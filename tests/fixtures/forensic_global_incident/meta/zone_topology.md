# Infrastructure Topology - Q1 2026

## Zone A (Primary)
- Gateway: GW-ALPHA-01
- Load Balancer: LB-PROD-01
- Clusters:
  - Auth-V2 (Nodes: 12)
  - Payment-Worker (Nodes: 45)

## Zone B (High-IO)
- Gateway: GW-BRAVO-02
- Clusters:
  - Titan-Primary (Database: PostgreSQL High-Availability)
  - Cache-Layer (Redis-6.2)
  - Forensic-Store (ClickHouse)

## Zone C (APAC)
- Gateway: GW-CHARLIE-03
- Clusters:
  - Titan-Read-Replica-01
  - User-Profile-Store
