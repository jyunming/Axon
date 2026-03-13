# System Architecture
The E-Commerce platform consists of three main microservices:
1. Service A (Auth): Handles user logins. Connects to `Users` table.
2. Service B (Catalog): Handles product listings. Connects to `Inventory` table.
3. Service C (Payments): Handles checkout and billing. Connects to `Transactions` table.

All microservices are routed through the API Gateway. If Service C fails to respond within 500ms, the API Gateway returns a 504 Gateway Timeout error to the client, effectively failing the checkout process.
