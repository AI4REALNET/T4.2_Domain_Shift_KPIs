#!/bin/bash
docker run -it --rm --name ds-kpi \
-v ./config.ini:/usr/src/config.ini \
-v ./test_results:/usr/src/results mleyliabadi/ds-kpi:v3