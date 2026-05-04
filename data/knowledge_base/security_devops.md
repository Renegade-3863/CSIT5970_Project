# Security Knowledge Base

## OWASP Top 10 (2021)

### A01: Broken Access Control
- Most critical OWASP category in 2021 (moved from #5)
- Includes BOLA (Broken Object Level Authorization) / IDOR
- API impact: attackers access other users' data by changing object IDs
- Mitigation: enforce access control server-side, deny by default

### A02: Cryptographic Failures
- Formerly "Sensitive Data Exposure"  
- Weak algorithms (MD5, SHA1, DES), missing encryption in transit
- Mitigation: TLS 1.3, AES-256-GCM, bcrypt/Argon2 for passwords

### A03: Injection
- SQL, NoSQL, LDAP, command injection
- Highest API impact via parameter manipulation
- Mitigation: parameterized queries, prepared statements, input validation

### A04: Insecure Design
- New in 2021 - design-level flaws
- Missing threat modeling, insecure design patterns
- Mitigation: threat modeling (STRIDE), security design reviews

### A05: Security Misconfiguration
- Default credentials, unnecessary features enabled, verbose error messages
- Cloud misconfigurations (open S3 buckets, permissive IAM)
- Mitigation: hardening guides (CIS benchmarks), automated config scanning

### A06: Vulnerable and Outdated Components
- Using components with known CVEs
- Mitigation: SCA tools (Snyk, Dependabot), patch management

### A07: Identification and Authentication Failures
- Broken auth, weak session management, credential stuffing
- API tokens not expired, JWT algorithm confusion
- Mitigation: MFA, strong session tokens, rate limiting login

### A08: Software and Data Integrity Failures
- Insecure CI/CD, unsigned packages, auto-update without integrity checks
- Log4Shell was a supply chain attack example
- Mitigation: SBOMs, code signing, verified update channels

### A09: Security Logging and Monitoring Failures
- Insufficient logging of API abuse is critical for APIs
- No alerting on authentication failures or access violations
- Mitigation: structured logging, SIEM integration, anomaly detection

### A10: Server-Side Request Forgery (SSRF)
- Server fetches attacker-controlled URL, accesses internal resources
- Cloud metadata service attacks (169.254.169.254)
- Mitigation: allowlist URLs, disable unnecessary URL fetching

---

## Incident Response Playbooks

### Container Escape Response Playbook

**Severity: Critical**

#### Phase 1: Immediate Containment (0-15 minutes)
1. Cordon affected Kubernetes nodes: `kubectl cordon <node-name>`
2. Drain workloads: `kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data`
3. Isolate nodes at network level (security group / network policy)
4. Do NOT reboot yet - preserve volatile evidence

#### Phase 2: Evidence Collection (15-60 minutes)
1. Capture running processes: `ps auxef > /tmp/processes.txt`
2. Capture network connections: `ss -tupn > /tmp/connections.txt`
3. Container runtime audit logs: `/var/log/containerd/` or `/var/log/docker/`
4. Kernel audit logs: `ausearch -k container_escape`
5. Memory capture if possible (LiME module)
6. Falco alert logs: `/var/log/falco.log`

#### Phase 3: Blast Radius Assessment
1. Identify which pods ran on compromised nodes
2. Review service account permissions: `kubectl get rolebindings,clusterrolebindings -A`
3. Check for lateral movement indicators in logs
4. Review etcd for unauthorized changes

#### Phase 4: Remediation
1. Patch container runtime (containerd/runc) to latest version
2. Apply kernel security patches
3. Re-enable nodes after patching: `kubectl uncordon <node-name>`

#### Phase 5: Post-Incident Hardening
1. Implement Falco rules for container escape detection
2. Enable seccomp profiles for all pods (RuntimeDefault)
3. Enable AppArmor profiles
4. Review and restrict privileged pod usage
5. Implement Pod Security Standards (restricted profile)
6. Add OPA Gatekeeper policies

---

## Kubernetes Security Best Practices

### RBAC (Role-Based Access Control)
- Enable RBAC: `--authorization-mode=RBAC`
- Apply least-privilege: start with no permissions, add as needed
- Avoid ClusterAdmin for regular workloads
- Regular RBAC audits: `kubectl auth can-i --list --as=system:serviceaccount:default:default`

### Network Policies
Default deny all ingress/egress, then allowlist:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

### Pod Security Standards
Apply via namespace labels:
```yaml
labels:
  pod-security.kubernetes.io/enforce: restricted
  pod-security.kubernetes.io/audit: restricted
  pod-security.kubernetes.io/warn: restricted
```

### Secrets Management
- Never store secrets in ConfigMaps or environment variables
- Use External Secrets Operator with HashiCorp Vault or AWS Secrets Manager
- Enable etcd encryption at rest: `--encryption-provider-config`
- Disable service account token auto-mounting: `automountServiceAccountToken: false`

### Supply Chain Security
- Sign container images with cosign (Sigstore)
- Verify signatures in admission controller (Kyverno)
- Generate and verify SBOMs (CycloneDX/SPDX format)
- Scan images in registry: Trivy, Grype, or Snyk Container

### Audit Logging
Enable comprehensive audit logging in kube-apiserver:
```yaml
--audit-log-path=/var/log/kubernetes/audit.log
--audit-log-maxage=30
--audit-log-maxbackup=10
--audit-policy-file=/etc/kubernetes/audit-policy.yaml
```

---

## Zero-Downtime Database Migration Procedure

### The Expand/Contract Pattern

Recommended for high-traffic production systems:

#### Phase 1: Expand (Non-breaking changes)
1. Add new columns/tables (nullable, no constraints initially)
2. Deploy code that writes to BOTH old and new schema
3. Run verification queries to ensure double-writes succeed

#### Phase 2: Backfill
1. Backfill existing data in batches to avoid table locks:
```sql
UPDATE users 
SET new_column = compute_value(old_column)
WHERE id BETWEEN :start AND :end
  AND new_column IS NULL;
```
2. Use `pg_repack` for PostgreSQL to avoid locking
3. Use `gh-ost` for MySQL/MariaDB online schema changes

#### Phase 3: Migrate Reads
1. Deploy code that reads from new schema
2. Use feature flags for gradual rollout
3. Monitor error rates and latency

#### Phase 4: Contract (Cleanup)
1. Verify old columns no longer used
2. Remove old columns in a separate deployment
3. Add any deferred constraints (NOT NULL, foreign keys)

### Tools
- **PostgreSQL**: pg_repack, pglogical for replication-based migration
- **MySQL**: gh-ost (GitHub Online Schema Transmogrifier), pt-online-schema-change
- **Feature flags**: LaunchDarkly, Unleash for gradual cutover
- **Monitoring**: Track query performance during migration with pg_stat_statements
