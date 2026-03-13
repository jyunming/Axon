# Mega-Complex Use Case: The Region West Blackout

This test case is designed to verify 5-hop forensic reasoning across multiple file formats.

## The Challenge
Find the root cause of the payment failure in **Region West** on 2026-03-13.

## Expected Answer
The failure was caused by an **automated cleanup bot (`AUTO_CLEANUP_BOT`)** blocking **Port 9921** in **Zone B** at **14:00 UTC**.
This port belongs to the **Titan-Primary database**, which serves all traffic for **Region West**.
The bot incorrectly identified the port as "unused" due to stale metadata.

## Logical Trace (The 5 Hops)
1. **Region West** maps to **Zone B** (`meta/routing_table.tsv`).
2. **Zone B** contains the **Titan-Primary** cluster (`meta/zone_topology.md`).
3. **Titan-Primary** runs on **Port 9921** (`config/db_inventory.json`).
4. **Port 9921** was blocked at **14:00** in **Zone B** (`logs/firewall_audit.csv`).
5. **Connection timeouts** began at **14:05** for that specific database (`logs/service_logs.txt`).

## Files Created:
- `meta/routing_table.tsv` (Tab-separated)
- `meta/zone_topology.md` (Markdown)
- `config/db_inventory.json` (JSON)
- `logs/firewall_audit.csv` (Ragged CSV with extra trailing columns)
- `logs/service_logs.txt` (Plain text logs)
