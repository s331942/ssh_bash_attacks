jq --indent 1 \
    '
    (.cells[] | select(has("outputs")) | .outputs) = []
    | (.cells[] | select(has("execution_count")) | .execution_count) = null
    | .metadata = {"language_info": {"name":"python", "pygments_lexer": "ipython3"}}
    | .cells[].metadata = {}
    ' section1_DataExploration_and_preprocessing.ipynb > section1_DataExploration_and_preprocessing_stripped.ipynb && mv section1_DataExploration_and_preprocessing_stripped.ipynb section1_DataExploration_and_preprocessing.ipynb