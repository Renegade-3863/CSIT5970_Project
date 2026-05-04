# Linux System Administration Knowledge Base

## Kernel Parameter Tuning for PostgreSQL

For high-performance PostgreSQL on Linux, the following sysctl settings are recommended:

### Virtual Memory Settings
```
vm.swappiness = 1               # Minimize swapping; 0 may cause OOM issues
vm.dirty_ratio = 10             # Max percentage of RAM for dirty pages before process blocks
vm.dirty_background_ratio = 5   # Percentage at which background flush starts
vm.overcommit_memory = 2        # Prevent overcommit (good for DB stability)
vm.overcommit_ratio = 80        # Allow up to 80% of RAM + swap to be committed
vm.nr_hugepages = 1024          # Enable huge pages (match shared_buffers in postgresql.conf)
```

### Shared Memory Settings
```
kernel.shmmax = 68719476736     # Maximum size of a single shared memory segment (64GB)
kernel.shmall = 4294967296      # Total pages of shared memory system-wide
```

### Network Settings for High Connections
```
net.core.somaxconn = 65535      # Maximum connection backlog
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 65535
```

### Transparent Hugepages (MUST disable for PostgreSQL)
```bash
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag
```
Add to /etc/rc.local for persistence.

### File Descriptor Limits
Edit /etc/security/limits.conf:
```
postgres soft nofile 65536
postgres hard nofile 65536
```

### Apply sysctl settings persistently
Edit /etc/sysctl.d/99-postgresql.conf and run `sysctl -p /etc/sysctl.d/99-postgresql.conf`.

---

## Log Rotation Best Practices

Configure logrotate for application logs in /etc/logrotate.d/:

```
/var/log/application/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 appuser appgroup
    sharedscripts
    postrotate
        kill -HUP $(cat /var/run/application.pid 2>/dev/null) 2>/dev/null || true
    endscript
}
```

Key directives:
- `compress`: gzip completed logs
- `delaycompress`: compress the previous rotation (not the most recent)  
- `notifempty`: skip rotation if log is empty
- `missingok`: don't error if log doesn't exist
- `sharedscripts`: run postrotate once even if multiple logs match

---

## System Performance Monitoring Commands

### CPU Analysis
```bash
# Real-time CPU usage by process
top -b -n 1 | head -20

# CPU frequency and core info  
lscpu
cat /proc/cpuinfo

# Context switches and interrupts
vmstat 1 5

# Per-CPU statistics
mpstat -P ALL 1 5
```

### Memory Analysis
```bash
# Memory usage summary
free -h

# Detailed memory info
cat /proc/meminfo

# Top memory consumers
ps auxf | sort -rnk 4 | head -10

# NUMA memory statistics
numastat
```

### Disk I/O Analysis
```bash
# Disk utilization
iostat -xz 1 5

# Top I/O processes
iotop -o

# Find large files
find / -type f -size +100M -exec ls -lh {} \; 2>/dev/null | sort -rk5
```

### Network Analysis
```bash
# Active connections
ss -tulpn
netstat -tulpn  

# Network throughput
iftop -i eth0
nethogs

# Packet statistics
ip -s link show eth0
```
