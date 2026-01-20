# Code Retrieval Example

This project is a demonstration of a code retrieval system defined in ***plain.

The main ***plain specification file is the: `code_retrieval_cli.plain`.

The unit test and conformance test scripts are defined in `run_unittests_python.sh` and `run_conformance_tests_python.sh`, respectively. They are both already set in the `config.yaml` file that is used by default by the plain2code renderer.

## Usage

Once rendered, the main entry point for this example is `code_search.py` located in the `build` folder.

First, navigate to the `build` directory:
```bash
cd build
```

Here are some of the available commands:

### Show Help

To see all available options, run:
```bash
python code_search.py --help
```

### Initialize the Index

To initialize the index with the a folder located at $FOLDER_PATH, run the following command. This will create the search index from the specified folder.
```bash
python code_search.py init $FOLDER_PATH
```

### Query the Index

You can also perform queries on the index in natural language. For example:
```bash
python code_search.py query code_index.faiss code_metadata.pkl "Some natural language query"
```

### Add an additional folder to the index
To add an additional folder at the $FOLDER_PATH location into the index you can do:

```bash
python code_search.py add-folder $FOLDER_PATH code_index.faiss code_metadata.pkl
```