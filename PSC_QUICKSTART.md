# PSC Automation - Quick Reference

Your one-command controller for PSC Bridges-2 experiments.

---

## Setup (One Time)

```bash
cd ~/Downloads/LLMsys_Project/vllm
chmod +x psc_controller.sh

# Test connection
./psc_controller.sh status
```

---

## Most Common Commands

### Deploy Method 2 (Recommended - One Command)

```bash
./psc_controller.sh quick
```

This does everything:
1. Syncs all files to PSC
2. Submits Method 2 job
3. Shows how to monitor

---

### Manual Workflow

```bash
# 1. Sync files to PSC
./psc_controller.sh sync

# 2. Submit a job
./psc_controller.sh submit method2

# 3. Check status
./psc_controller.sh status

# 4. Watch live output
./psc_controller.sh watch

# 5. Download results
./psc_controller.sh results
```

---

## All Commands

| Command | What It Does |
|---------|-------------|
| `./psc_controller.sh quick` | **ONE-COMMAND: Sync + submit Method 2** |
| `./psc_controller.sh sync` | Sync all test files to PSC |
| `./psc_controller.sh submit method1` | Submit Method 1 (Batch Inference) |
| `./psc_controller.sh submit method2` | Submit Method 2 (Custom Serving) |
| `./psc_controller.sh submit all` | Submit both methods |
| `./psc_controller.sh status` | Check job status |
| `./psc_controller.sh watch` | Watch latest job output (live) |
| `./psc_controller.sh watch 38686568` | Watch specific job ID |
| `./psc_controller.sh results` | Download latest result file |
| `./psc_controller.sh download_all` | Download all result files |
| `./psc_controller.sh clean` | Delete old output files on PSC |
| `./psc_controller.sh help` | Show full help |

---

## Common Workflows

### Workflow 1: Submit Method 2 and Monitor

```bash
# One command
./psc_controller.sh quick

# Or manually
./psc_controller.sh sync
./psc_controller.sh submit method2
./psc_controller.sh watch
```

### Workflow 2: Check Results

```bash
# Download latest
./psc_controller.sh results

# Or download everything
./psc_controller.sh download_all

# Then check eviction counts
grep "Total KV Cache Block Evictions" week1_*.out
```

### Workflow 3: Submit Multiple Methods

```bash
./psc_controller.sh sync
./psc_controller.sh submit all

# Check both
./psc_controller.sh status
```

---

## Configuration

Your settings (in `psc_controller.sh`):
- **PSC User**: rnagaraj
- **PSC Host**: bridges2.psc.edu
- **PSC Account**: cis250224p
- **Workspace**: ~/workspace/vllm
- **Local Path**: ~/Downloads/LLMsys_Project/vllm

All configured and ready to use!

---

## What Each Method Tests

### Method 1: Batch Inference
- Standard vLLM LLM API
- 80% GPU memory
- 20 prompts × 10K tokens = 300K tokens
- **Status**: FAILED (0 evictions)

### Method 2: Custom Serving (CURRENT)
- AsyncLLMEngine direct access
- **35% GPU memory** (tighter!)
- **16 concurrent requests** via asyncio.gather
- 16 prompts × 8K tokens = 128K tokens
- **Goal**: >1000 evictions

---

## Success Criteria

When you run `./psc_controller.sh results`, look for:

```
Total KV Cache Block Evictions: XXXXX
```

- **✅ >1000**: SUCCESS! Start Week 2 LongBench
- **⚠️ 100-1000**: PARTIAL, try Method 3 (GCP)
- **❌ <100**: FAILURE, pivot to Path B (workshop paper)

---

## Troubleshooting

### Connection Issues
```bash
# Test SSH
ssh rnagaraj@bridges2.psc.edu

# If prompted for password, add SSH key:
ssh-copy-id rnagaraj@bridges2.psc.edu
```

### No Files Found
```bash
# Check what's on PSC
ssh rnagaraj@bridges2.psc.edu "ls -la ~/workspace/vllm/week1_*.out"
```

### Job Not Starting
```bash
# Check queue
./psc_controller.sh status

# Check account allocation
ssh rnagaraj@bridges2.psc.edu "sacctmgr show assoc user=rnagaraj"
```

---

## Quick Reference Card

```bash
# MOST COMMON (copy-paste this)
cd ~/Downloads/LLMsys_Project/vllm
./psc_controller.sh quick          # Deploy Method 2
./psc_controller.sh status         # Check status
./psc_controller.sh watch          # Watch live
./psc_controller.sh results        # Get results
```

**Bookmark this file!** 📌
