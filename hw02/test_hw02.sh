#!/bin/bash

# Test script for hw02.py GCN Link Prediction
# This script runs multiple parameter combinations and logs the best results

LOG_FILE="log.txt"
PYTHON_SCRIPT="hw02.py"

# Clear previous log
> $LOG_FILE

echo "GCN Link Prediction Parameter Testing" >> $LOG_FILE
echo "=====================================" >> $LOG_FILE
echo "Started: $(date)" >> $LOG_FILE
echo "" >> $LOG_FILE

# Parameter combinations to test
HIDDEN_DIMS=(16 32 64 128 256)
LEARNING_RATES=(0.01 0.005 0.001) 
DROPOUT_RATES=(0.1 0.2 0.3 0.4 0.5)
EPOCHS=30
PATIENCE=5

echo "Testing parameter combinations..." | tee -a $LOG_FILE

for hidden in "${HIDDEN_DIMS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for dropout in "${DROPOUT_RATES[@]}"; do
            echo "Running: hidden=$hidden, lr=$lr, dropout=$dropout" | tee -a $LOG_FILE
            
            # Run the script and capture output
            output=$(python $PYTHON_SCRIPT \
                --epochs $EPOCHS \
                --lr $lr \
                --hidden $hidden \
                --dropout $dropout \
                --patience $PATIENCE \
                2>&1)
            
            # Extract best validation AUC and test AUC
            best_val=$(echo "$output" | grep "Best validation AUC" | awk '{print $5}')
            test_auc=$(echo "$output" | grep "Test AUC" | awk '{print $4}')
            
            # Log the results
            echo "  Best Validation AUC: $best_val" >> $LOG_FILE
            echo "  Test AUC: $test_auc" >> $LOG_FILE
            echo "  ---" >> $LOG_FILE
            
            echo "  Completed: Val AUC=$best_val, Test AUC=$test_auc"
        done
    done
done

echo "" >> $LOG_FILE
echo "Testing completed: $(date)" >> $LOG_FILE

# Find and log the best overall results
echo "" >> $LOG_FILE
echo "SUMMARY - Best Results:" >> $LOG_FILE
echo "======================" >> $LOG_FILE

best_val_overall=$(grep "Best Validation AUC:" $LOG_FILE | awk '{print $4}' | sort -rn | head -1)
best_test_overall=$(grep "Test AUC:" $LOG_FILE | awk '{print $3}' | sort -rn | head -1)

echo "Best Validation AUC across all runs: $best_val_overall" >> $LOG_FILE
echo "Best Test AUC across all runs: $best_test_overall" >> $LOG_FILE

echo "All results saved to $LOG_FILE"