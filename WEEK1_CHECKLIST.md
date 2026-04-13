# Week 1 Checklist: Force Production Evictions

**Goal**: Achieve >1000 evictions in production benchmark to unlock Path A (MLSys main conference)

**Timeline**: Monday-Thursday (April 7-10, 2026)

---

## Monday Morning: Method 1 - Batch Inference

### 1. Sync Files to PSC

```bash
cd ~/Downloads/LLMsys_Project/vllm

rsync -av test_bypass_scheduler.py \
    bridges2.psc.edu:~/workspace/vllm/
```

### 2. Create SLURM Script on PSC

```bash
ssh bridges2.psc.edu
cd ~/workspace/vllm

cat > slurm_week1_method1.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=week1_method1
#SBATCH --output=week1_method1_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --account=cis250224p

set -e

echo "Week 1 - Method 1: Batch Inference with Long Sequences"
echo "Start time: $(date)"

source ~/workspace/vllm/.venv/bin/activate
cd ~/workspace/vllm

python test_bypass_scheduler.py

echo "End time: $(date)"
EOF

chmod +x slurm_week1_method1.sh
```

### 3. Submit and Monitor

```bash
sbatch slurm_week1_method1.sh

# Monitor progress
squeue -u $USER
tail -f week1_method1_*.out
```

### 4. Check Results

Look for output:
```
Total evictions: XXXXX
```

**Success**: evictions > 1000 → ✅ STOP, proceed to Week 2 planning
**Partial**: evictions 100-1000 → ⚠️ Try Method 2
**Failure**: evictions < 100 → ❌ Try Method 2

---

## Tuesday: Method 2 - Custom Serving (if Method 1 failed)

### 1. Sync Method 2

```bash
rsync -av test_custom_serving.py \
    bridges2.psc.edu:~/workspace/vllm/
```

### 2. Create SLURM Script

```bash
ssh bridges2.psc.edu
cd ~/workspace/vllm

cat > slurm_week1_method2.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=week1_method2
#SBATCH --output=week1_method2_%j.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --account=cis250224p

set -e

echo "Week 1 - Method 2: Custom Serving with High Concurrency"
echo "Start time: $(date)"

source ~/workspace/vllm/.venv/bin/activate
cd ~/workspace/vllm

python test_custom_serving.py

echo "End time: $(date)"
EOF

chmod +x slurm_week1_method2.sh
```

### 3. Submit and Monitor

```bash
sbatch slurm_week1_method2.sh
squeue -u $USER
tail -f week1_method2_*.out
```

### 4. Check Results

**Success**: evictions > 1000 → ✅ Proceed to Week 2
**Failure**: evictions < 100 → ❌ Try Method 3 (GCP)

---

## Wednesday: Method 3 - GCP Fresh Environment (if Methods 1-2 failed)

### 1. Create GCP V100 Instance

```bash
gcloud compute instances create vllm-week1-test \
  --zone=us-central1-a \
  --machine-type=n1-highmem-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --preemptible \
  --boot-disk-size=100GB
```

### 2. Install vLLM

```bash
gcloud compute ssh vllm-week1-test --zone=us-central1-a

# On GCP instance
sudo apt-get update
sudo apt-get install -y python3-pip git

# Clone and install vLLM
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .

# Copy test scripts (from local)
# On local machine:
gcloud compute scp test_bypass_scheduler.py \
  vllm-week1-test:~/vllm/ --zone=us-central1-a
```

### 3. Run Test

```bash
# On GCP instance
cd ~/vllm
python test_bypass_scheduler.py
```

### 4. Check Results

**Success**: evictions > 1000 → ✅ Week 2 on GCP
**Failure**: evictions < 100 → ❌ Pivot to Path B

### 5. Clean Up

```bash
# Delete instance to save money
gcloud compute instances delete vllm-week1-test --zone=us-central1-a
```

---

## Thursday: GO/NO-GO Decision

### If ANY method achieved >1000 evictions:

**✅ GO - Commit to Path A**

1. **Plan Week 2** (LongBench evaluation):
   - Download datasets
   - Prepare automation scripts
   - Reserve GPU time/budget

2. **Block calendar**:
   - 40 hours/week for next 5 weeks
   - No major breaks

3. **Confirm budget**:
   - $110 for Weeks 2-6
   - Payment method ready

4. **Create Week 2 plan**:
   - LongBench tasks setup
   - Model downloads
   - Experiment scheduling

### If ALL methods failed (<100 evictions):

**❌ NO-GO - Pivot to Path B**

1. **Accept reality**:
   - V1 scheduler is too conservative
   - Unit tests are strong on their own
   - Workshop paper is still valuable

2. **Start Path B immediately**:
   - Create graphs from unit test data (Friday)
   - Outline workshop paper (next week)
   - Research workshop deadlines

3. **Reframe contribution**:
   - "Rigorous controlled experiments..."
   - "Novel attention-aware eviction policy..."
   - "One production validation (ShareGPT 9.2%)..."

4. **Set new timeline**:
   - 2 weeks: Create figures + outline
   - 4 weeks: Write + revise
   - Submit: June 2026 workshops

---

## Progress Tracking

| Day | Task | Status | Evictions | Decision |
|-----|------|--------|-----------|----------|
| Mon | Method 1: Batch inference | ⬜ | ? | ? |
| Tue | Method 2: Custom serving | ⬜ | ? | ? |
| Wed | Method 3: GCP fresh env | ⬜ | ? | ? |
| Thu | GO/NO-GO decision | ⬜ | ? | ? |

**Fill in as you complete each step.**

---

## Success Criteria

**Week 1 SUCCESS** = Any method achieves >1000 evictions
**Week 1 FAILURE** = All methods achieve <100 evictions

**If borderline (100-1000 evictions)**: Try one more variation:
- Longer sequences (20K tokens)
- Higher concurrency (32 requests)
- Different model (Llama-3.2-7B instead of Qwen)

---

## Quick Reference: What to Look For in Output

### ✅ SUCCESS Output:
```
======================================================================
RESULTS
======================================================================
Total evictions: 2847
Success threshold: >1000 evictions
✅ SUCCESS: Ready for Week 2 (LongBench)
   Proceed with production evaluation
======================================================================
```

### ⚠️ PARTIAL Output:
```
Total evictions: 342
Success threshold: >1000 evictions
⚠️ PARTIAL: Some evictions, but below threshold
   Try Method 2 (custom_serving.py) with higher concurrency
```

### ❌ FAILURE Output:
```
Total evictions: 0
Success threshold: >1000 evictions
❌ FAILURE: Zero evictions
   V1 scheduler still blocking - consider Path B
```

---

## Contact/Questions

If you get stuck:
1. Check SLURM output files for errors
2. Verify vLLM version: `python -c "import vllm; print(vllm.__version__)"`
3. Check GPU memory: `nvidia-smi`
4. Review test script parameters (gpu_memory_utilization, max_model_len)

**Ready to start Monday morning?** 🚀

Good luck! This is the critical week that determines your publication path.
