#!/bin/bash
docker run -it --rm --name ds-kpi \
-v ./config.ini:/usr/src/config.ini \
-v /home/milad/repos/ai4realnet/grid2op-scenario/l2rpn_case14_sandbox:/root/data_grid2op/l2rpn_case14_sandbox \
-v ./submission:/usr/src/submission \
-v ./test_results:/usr/src/results mleyliabadi/ds-kpi:v2