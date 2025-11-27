#!/bin/bash
# Parameter sweep script for sim2 Gillespie simulations
#
# Usage examples:
#   ./run_sweep.sh copies        # Sweep gene copy numbers
#   ./run_sweep.sh mu            # Sweep mutation rates
#   ./run_sweep.sh s             # Sweep selection coefficients
#   ./run_sweep.sh N             # Sweep carrying capacities
#   ./run_sweep.sh grid          # Full parameter grid search
#   ./run_sweep.sh quick_test    # Quick test with minimal runs

set -e  # Exit on error

# Default parameters
NUM_RUNS=20
START_WORD="WORD"
TARGET_WORD="GENE"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}Gillespie Simulation Parameter Sweep${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Parse command line argument
SWEEP_TYPE=${1:-copies}

case $SWEEP_TYPE in
    copies)
        echo -e "${GREEN}Running gene copy number sweep...${NC}"
        echo -e "Parameters: ${NUM_RUNS} runs per condition"
        echo ""
        python sim2_stats.py sweep_copies \
            --num_runs=$NUM_RUNS \
            --start=$START_WORD \
            --target=$TARGET_WORD \
            --max_copies=5 \
            --mu=0.02 \
            --s=0.3 \
            --N=1000
        ;;

    mu)
        echo -e "${GREEN}Running mutation rate sweep...${NC}"
        echo -e "Parameters: ${NUM_RUNS} runs per condition"
        echo ""
        python sim2_stats.py sweep_mu \
            --num_runs=$NUM_RUNS \
            --start=$START_WORD \
            --target=$TARGET_WORD \
            --n_copies=2 \
            --mu_values="0.005,0.01,0.02,0.05,0.1" \
            --s=0.3 \
            --N=1000
        ;;

    s)
        echo -e "${GREEN}Running selection coefficient sweep...${NC}"
        echo -e "Parameters: ${NUM_RUNS} runs per condition"
        echo ""
        python sim2_stats.py sweep_s \
            --num_runs=$NUM_RUNS \
            --start=$START_WORD \
            --target=$TARGET_WORD \
            --n_copies=2 \
            --s_values="0.1,0.2,0.5,1.0,2.0" \
            --mu=0.02 \
            --N=1000
        ;;

    N)
        echo -e "${GREEN}Running carrying capacity sweep...${NC}"
        echo -e "Parameters: ${NUM_RUNS} runs per condition"
        echo ""
        python sim2_stats.py sweep_N \
            --num_runs=$NUM_RUNS \
            --start=$START_WORD \
            --target=$TARGET_WORD \
            --n_copies=2 \
            --N_values="100,500,1000,2000,5000" \
            --mu=0.02 \
            --s=0.3
        ;;

    grid)
        echo -e "${GREEN}Running full parameter grid search...${NC}"
        echo -e "${YELLOW}WARNING: This will run many simulations!${NC}"
        echo ""
        python sim2_stats.py sweep_grid \
            --num_runs=10 \
            --start=$START_WORD \
            --target=$TARGET_WORD \
            --n_copies_values="1,2,3" \
            --mu_values="0.01,0.02,0.05" \
            --s_values="0.2,0.5,1.0" \
            --N_values="500,1000,2000"
        ;;

    quick_test)
        echo -e "${GREEN}Running quick test (3 runs, 3 copy numbers)...${NC}"
        echo ""
        python sim2_stats.py sweep_copies \
            --num_runs=3 \
            --start=$START_WORD \
            --target=$TARGET_WORD \
            --max_copies=3 \
            --mu=0.02 \
            --s=0.3 \
            --N=1000
        ;;

    custom)
        echo -e "${YELLOW}Custom sweep - modify this script to add your parameters${NC}"
        echo ""
        # Example custom sweep - edit as needed
        python sim2_stats.py sweep_copies \
            --num_runs=10 \
            --start="COLD" \
            --target="WARM" \
            --max_copies=4 \
            --mu=0.015 \
            --s=0.4 \
            --N=1500
        ;;

    *)
        echo -e "${RED}Error: Unknown sweep type '$SWEEP_TYPE'${NC}"
        echo ""
        echo "Available sweep types:"
        echo "  copies      - Sweep gene copy numbers (1-5)"
        echo "  mu          - Sweep mutation rates"
        echo "  s           - Sweep selection coefficients"
        echo "  N           - Sweep carrying capacities"
        echo "  grid        - Full parameter grid search"
        echo "  quick_test  - Quick test with minimal runs"
        echo "  custom      - Custom parameters (edit script)"
        echo ""
        echo "Usage: $0 [sweep_type]"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}Sweep completed successfully!${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo "Results saved to: results/sim2/"
echo ""
