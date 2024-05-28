#!/bin/bash
SCRIPT_PATH="run.py"

# Constant parameters
CONFIG="configs/run.json"
MODEL="run_short_form"
DATASET="triviaqa"
TASK="triviaqa"
MAX_NEW_TOKENS=1024
METRIC="match"

# triviaqa
python $SCRIPT_PATH \
--config $CONFIG \
--model $MODEL \
--dataset $DATASET \
--task $TASK \
--max_new_tokens $MAX_NEW_TOKENS \
--retrieve_method "bge_serper" \
--metric $METRIC \
--use_tvq

# popqa
#!/bin/bash
SCRIPT_PATH="run.py"

# Constant parameters
CONFIG="configs/run.json"
MODEL="run_short_form"
DATASET="popqa"
TASK="popqa"
MAX_NEW_TOKENS=1024
METRIC="match"

# triviaqa
python $SCRIPT_PATH \
--config $CONFIG \
--model $MODEL \
--dataset $DATASET \
--task $TASK \
--max_new_tokens $MAX_NEW_TOKENS \
--retrieve_method "bge_serper" \
--metric $METRIC \
--use_tvq \
--continue_gen_without_contents

# asqa
#!/bin/bash
SCRIPT_PATH="run.py"

# Constant parameters
CONFIG="configs/run.json"
MODEL="run_long_form"
DATASET="asqa"
TASK="asqa"
MAX_NEW_TOKENS=130

# asqa
python $SCRIPT_PATH \
--config $CONFIG \
--model $MODEL \
--dataset $DATASET \
--task $TASK \
--max_new_tokens $MAX_NEW_TOKENS \
--retrieve_method "bge" \
--use_tvq \

# bio
#!/bin/bash
SCRIPT_PATH="run.py"

# Constant parameters
CONFIG="configs/run.json"
MODEL="run_long_form"
DATASET="fact"
TASK="fact"
MAX_NEW_TOKENS=300

python $SCRIPT_PATH \
--config $CONFIG \
--model $MODEL \
--dataset $DATASET \
--task $TASK \
--max_new_tokens $MAX_NEW_TOKENS \
--retrieve_method "bge_serper" \
--use_tvq \

# freshqa
#!/bin/bash
SCRIPT_PATH="run.py"

# Constant parameters
CONFIG="configs/run.json"
MODEL="run_long_form"
DATASET="fresh"
TASK="fresh"
MAX_NEW_TOKENS=1024

python $SCRIPT_PATH \
--config $CONFIG \
--model $MODEL \
--dataset $DATASET \
--task $TASK \
--max_new_tokens $MAX_NEW_TOKENS \
--retrieve_method "serper" \
--use_tvq
