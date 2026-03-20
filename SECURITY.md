# Security Policy

## Supported Versions

Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.9.x   | :white_check_mark: |
| < 0.9   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

### Private Disclosure

**DO NOT** open a public issue for security vulnerabilities.

Instead, please report security issues via:
1. GitHub Security Advisories (preferred)
2. Email to the maintainers (if available)

### What to Include

When reporting a security vulnerability, please include:

- **Description**: Clear description of the vulnerability
- **Impact**: Potential impact and severity
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Proof of Concept**: If applicable, include PoC code
- **Affected Versions**: Which versions are affected
- **Suggested Fix**: If you have ideas for fixing it

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 5 business days
- **Status Updates**: Weekly until resolved
- **Fix Timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 30 days
  - Medium: Within 90 days

## Security Best Practices

### When Using Axon

1. **Input Validation**
   - Sanitize file paths before ingestion
   - Validate configuration files
   - Don't ingest untrusted files without scanning

2. **API Security**
   - Run the API behind a reverse proxy (nginx, traefik)
   - Use authentication/authorization if exposing publicly
   - Rate limit API endpoints
   - Use HTTPS in production

3. **Data Privacy**
   - Keep vector stores and indexes on encrypted storage
   - Don't expose data directories via web server
   - Be cautious with image captioning (VLM may leak info in logs)

4. **Docker Security**
   - Don't run containers as root
   - Use read-only volumes where possible
   - Limit resource usage
   - Keep base images updated

5. **Dependencies**
   - Regularly update dependencies
   - Review security advisories
   - Use `pip-audit` or similar tools

### Common Vulnerabilities to Avoid

#### Path Traversal
```python
# BAD - vulnerable to path traversal
file_path = user_input
with open(file_path) as f:
    content = f.read()

# GOOD - validate and sanitize
from pathlib import Path
file_path = Path(base_dir) / user_input
if not file_path.resolve().is_relative_to(base_dir):
    raise ValueError("Invalid path")
```

#### Code Injection
```python
# BAD - vulnerable to code injection
eval(user_input)
exec(user_input)

# GOOD - use safe alternatives
import ast
result = ast.literal_eval(user_input)  # Only for literals
```

#### SQL Injection
```python
# BAD - vulnerable to SQL injection
query = f"SELECT * FROM docs WHERE id = '{user_id}'"

# GOOD - use parameterized queries
query = "SELECT * FROM docs WHERE id = ?"
cursor.execute(query, (user_id,))
```

## Security Features

### Current Protections
- Input sanitization in loaders
- Path validation for file operations
- YAML safe loading (no arbitrary code execution)
- Dependency pinning with version ranges
- **Optional API key authentication** — set the `RAG_API_KEY` environment variable to enable. When set, every request to `axon-api` must include the header `X-API-Key: <your-value>`. Axon does not generate the key; you choose and set it yourself (treat it like a password). Unset by default — suitable for local/LAN use without auth.

### Planned Improvements
- [ ] Implement rate limiting
- [ ] Add input validation middleware
- [ ] Security audit of file operations
- [ ] Automated dependency scanning in CI
- [ ] SBOM (Software Bill of Materials) generation

## Disclosure Policy

When a vulnerability is fixed:
1. We'll release a patched version
2. Publish a security advisory
3. Credit the reporter (if desired)
4. Update this document

## Security Checklist for Contributors

Before submitting code:
- [ ] No hardcoded secrets or credentials
- [ ] User input is validated and sanitized
- [ ] File operations use safe path handling
- [ ] No use of `eval()`, `exec()`, or similar dangerous functions
- [ ] Dependencies are pinned with version constraints
- [ ] Error messages don't leak sensitive information
- [ ] Sensitive data is not logged

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)

Thank you for helping keep Axon secure! 🔒
