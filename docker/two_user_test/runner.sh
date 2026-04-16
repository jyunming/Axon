#!/usr/bin/env bash
# Two-user share test runner.
# Run inside the `runner` container (depends_on alice + bob being healthy).
set -uo pipefail

OUT=/output
mkdir -p "$OUT"

ALICE_URL=http://alice:8000
BOB_URL=http://bob:8001

PASS=0
FAIL=0

# ── helpers ──────────────────────────────────────────────────────────────────

check() {
    local label="$1"
    local condition="$2"
    if eval "$condition"; then
        echo "  ✓ PASS  $label"
        PASS=$((PASS + 1))
    else
        echo "  ✗ FAIL  $label"
        FAIL=$((FAIL + 1))
    fi
}

jq_get() { python3 -c "import json,sys; d=json.load(open('$1')); print(d.get('$2',''))" 2>/dev/null; }
jq_len()  { python3 -c "import json,sys; d=json.load(open('$1')); print(len(d.get('$2',[])))" 2>/dev/null || echo 0; }
# Parse bare list OR {"results":[...]} from /search response
search_count() { python3 -c "import json; d=json.load(open('$1')); print(len(d) if isinstance(d,list) else len(d.get('results',[])))" 2>/dev/null || echo 0; }
search_text()  { python3 -c "import json; d=json.load(open('$1')); r=d if isinstance(d,list) else d.get('results',[]); print(r[0].get('text','') if r else '')" 2>/dev/null || echo ""; }

# ── Step 0: Cleanup from previous runs ───────────────────────────────────────
echo ""
echo "=== Step 0: Cleanup ==="
# Store init first (sets correct user paths) then delete stale sharedproj so
# content_hashes from previous runs don't block dedup on re-ingest.
curl -sS -X POST "$ALICE_URL/store/init" \
    -H "Content-Type: application/json" \
    -d '{"base_path":"/srv/axon-store","persist":false}' \
    -o /dev/null
# Switch to default before deleting sharedproj (can't delete active project)
curl -sS -X POST "$ALICE_URL/project/switch" \
    -H "Content-Type: application/json" \
    -d '{"name":"default"}' \
    -o /dev/null
curl -sS -X POST "$ALICE_URL/project/delete/sharedproj" \
    -H "Content-Type: application/json" \
    -d '{}' \
    -o /dev/null 2>/dev/null || true
echo "  Cleanup complete (any errors above are expected on first run)"

# ── Step 1: Store init ────────────────────────────────────────────────────────
echo ""
echo "=== Step 1: Store init ==="

curl -sS -X POST "$ALICE_URL/store/init" \
    -H "Content-Type: application/json" \
    -d '{"base_path":"/srv/axon-store","persist":false}' \
    -o "$OUT/alice_init.json"
check "Alice store init returns ok"      "[ \"\$(jq_get $OUT/alice_init.json status)\" = 'ok' ]"
check "Alice username is 'alice'"        "[ \"\$(jq_get $OUT/alice_init.json username)\" = 'alice' ]"

curl -sS -X POST "$BOB_URL/store/init" \
    -H "Content-Type: application/json" \
    -d '{"base_path":"/srv/axon-store","persist":false}' \
    -o "$OUT/bob_init.json"
check "Bob store init returns ok"        "[ \"\$(jq_get $OUT/bob_init.json status)\" = 'ok' ]"
check "Bob username is 'bob'"            "[ \"\$(jq_get $OUT/bob_init.json username)\" = 'bob' ]"

# ── Step 2: Alice creates project ─────────────────────────────────────────────
echo ""
echo "=== Step 2: Alice creates project ==="

curl -sS -X POST "$ALICE_URL/project/new" \
    -H "Content-Type: application/json" \
    -d '{"name":"sharedproj"}' \
    -o "$OUT/alice_proj.json"
alice_proj_status=$(jq_get "$OUT/alice_proj.json" status)
check "Alice project created" "[ '$alice_proj_status' = 'ok' ] || [ '$alice_proj_status' = 'success' ] || python3 -c \"import json; d=json.load(open('$OUT/alice_proj.json')); exit(0 if d.get('name')=='sharedproj' or d.get('status') in ('ok','success') else 1)\""

curl -sS -X POST "$ALICE_URL/project/switch" \
    -H "Content-Type: application/json" \
    -d '{"name":"sharedproj"}' \
    -o "$OUT/alice_switch.json"
check "Alice switched to sharedproj" "python3 -c \"import json; d=json.load(open('$OUT/alice_switch.json')); exit(0 if 'sharedproj' in str(d) else 1)\""

# ── Step 3: Alice ingests a document ─────────────────────────────────────────
echo ""
echo "=== Step 3: Alice ingests document ==="

curl -sS -X POST "$ALICE_URL/add_text" \
    -H "Content-Type: application/json" \
    -d '{"text":"Quantum entanglement is a physical phenomenon where two particles become correlated such that the quantum state of each cannot be described independently.","metadata":{"source":"alice_quantum.txt","topic":"physics"}}' \
    -o "$OUT/alice_ingest.json"
alice_ingest_id=$(jq_get "$OUT/alice_ingest.json" doc_id)
check "Alice ingest returns doc_id" "[ -n '$alice_ingest_id' ]"

# ── Step 4: Alice generates share key ────────────────────────────────────────
echo ""
echo "=== Step 4: Alice generates share key ==="

curl -sS -X POST "$ALICE_URL/share/generate" \
    -H "Content-Type: application/json" \
    -d '{"project":"sharedproj","grantee":"bob"}' \
    -o "$OUT/share_response.json"

share_string=$(jq_get "$OUT/share_response.json" share_string)
key_id=$(jq_get "$OUT/share_response.json" key_id)

check "Share key_id starts with sk_" "echo '$key_id' | grep -q '^sk_'"
check "Share string is non-empty"    "[ -n '$share_string' ]"
check "Share project is sharedproj"  "[ \"\$(jq_get $OUT/share_response.json project)\" = 'sharedproj' ]"
check "Share grantee is bob"         "[ \"\$(jq_get $OUT/share_response.json grantee)\" = 'bob' ]"

echo "  key_id: $key_id"

# ── Step 5: Bob redeems share ─────────────────────────────────────────────────
echo ""
echo "=== Step 5: Bob redeems share ==="

if [ -z "$share_string" ]; then
    echo "  ✗ FAIL  No share_string — skipping redeem steps"
    FAIL=$((FAIL + 1))
else
    curl -sS -X POST "$BOB_URL/share/redeem" \
        -H "Content-Type: application/json" \
        -d "{\"share_string\":\"$share_string\"}" \
        -o "$OUT/bob_redeem.json"

    mount_name=$(jq_get "$OUT/bob_redeem.json" mount_name)
    check "Bob redeem returns mount_name" "[ -n '$mount_name' ]"
    check "Mount name is alice_sharedproj" "[ '$mount_name' = 'alice_sharedproj' ]"
    check "Redeem owner is alice"          "[ \"\$(jq_get $OUT/bob_redeem.json owner)\" = 'alice' ]"

    echo "  mount_name: $mount_name"

    # ── Step 6: Bob lists shares ────────────────────────────────────────────────
    echo ""
    echo "=== Step 6: Bob lists shares ==="

    curl -sS "$BOB_URL/share/list" -o "$OUT/bob_share_list.json"
    received_count=$(jq_len "$OUT/bob_share_list.json" shared)
    check "Bob share list has ≥1 received entry" "[ '$received_count' -ge 1 ]"

    # ── Step 7: Bob switches to mounted project and searches ───────────────────
    echo ""
    echo "=== Step 7: Bob searches via mounted project ==="

    curl -sS -X POST "$BOB_URL/project/switch" \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"mounts/$mount_name\"}" \
        -o "$OUT/bob_switch.json"
    check "Bob switched to mounted project" "python3 -c \"import json; d=json.load(open('$OUT/bob_switch.json')); exit(0 if 'error' not in str(d).lower() or 'mounts/$mount_name' in str(d) else 1)\""

    curl -sS -X POST "$BOB_URL/search" \
        -H "Content-Type: application/json" \
        -d '{"query":"quantum entanglement","top_k":3}' \
        -o "$OUT/bob_search.json"

    result_count=$(search_count "$OUT/bob_search.json")
    check "Bob search returns ≥1 result from Alice's data" "[ '$result_count' -ge 1 ]"

    first_text=$(search_text "$OUT/bob_search.json")
    check "Search result contains 'quantum'" "echo '$first_text' | grep -qi 'quantum'"

    # ── Step 8: Alice revokes share ─────────────────────────────────────────────
    echo ""
    echo "=== Step 8: Alice revokes share ==="

    if [ -n "$key_id" ]; then
        curl -sS -X POST "$ALICE_URL/share/revoke" \
            -H "Content-Type: application/json" \
            -d "{\"key_id\":\"$key_id\"}" \
            -o "$OUT/alice_revoke.json"
        check "Revoke returns key_id" "[ \"\$(jq_get $OUT/alice_revoke.json key_id)\" = '$key_id' ]"

        sleep 2

        # ── Step 9: Bob's share list reflects revocation ────────────────────────
        echo ""
        echo "=== Step 9: Bob share list after revocation ==="

        curl -sS "$BOB_URL/share/list" -o "$OUT/bob_share_list_after_revoke.json"
        received_after=$(jq_len "$OUT/bob_share_list_after_revoke.json" shared)
        check "Bob received shares reduced after revoke" "[ '$received_after' -lt '$received_count' ] || python3 -c \"
import json
d=json.load(open('$OUT/bob_share_list_after_revoke.json'))
mounts=[r for r in d.get('shared',[]) if r.get('mount_name')=='$mount_name' and not r.get('revoked')]
exit(0 if len(mounts)==0 else 1)
\""
    else
        echo "  ✗ FAIL  No key_id — skipping revoke"
        FAIL=$((FAIL + 1))
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    echo "  OVERALL: FAIL"
    exit 1
else
    echo "  OVERALL: PASS"
    exit 0
fi

