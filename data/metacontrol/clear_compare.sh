ps aux | grep evaluate_diff.py | awk '{print $2}' | xargs kill -9
