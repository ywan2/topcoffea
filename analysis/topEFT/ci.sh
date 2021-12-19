work_queue_worker -d all -o worker.log localhost 9123 &
time python work_queue_run.py ../../topcoffea/json/test_samples/UL17_private_ttH_for_CI.json -o output_check_yields
