import os
import traceback
import pandas as pd
import logging

LOGGER_DIR_PATH='logs'

# Log Config
logFormatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

logger = logging.getLogger("__main__")
logger.setLevel(logging.DEBUG)
logger.handlers = []

fileHandler = logging.FileHandler(f"{LOGGER_DIR_PATH}/main.log")
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
def get_error_string(new_line=False):
    """
    Get the error string formatted from the traceback of the last exception.

    :param new_line: If True, preserve newlines; otherwise, replace them with tabs.
    :return: Formatted error string.
    """
    err_string = str(traceback.format_exc())
    if not new_line:
        err_string = err_string.replace('\n', '\t')
    return err_string


def append_row_to_csv(file_path, row, columns=None) -> None:
    """
    Write specific line to a .csv file located on disk.

    :param file_path: The path of the .csv file to append to.
    :param row: New record to be added.
    :param columns: .csv headers, optional parameter if row given as dictionary.
    """
    if columns is None:
        columns = list(row.keys())
    df = pd.DataFrame([row], columns=columns)
    file_exists = os.path.exists(file_path)
    if not file_exists:
        df.to_csv(file_path, header=True, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)

def log_message(logger_path, id_column, json_text, error="", request_text="", category="error"):
    """
    Log message to a specified CSV file.

    :param logger_path: Path to the CSV file for logging.
    :param id_column: Identifier column (e.g., ID or key).
    :param json_text: JSON representation to log (additional context).
    :param error: Error string to log.
    :param request_text: Request or relevant context text to log.
    :param category: "error" or "debug" to specify logging type.
    """
    if category == "error":
        file_path = os.path.join(logger_path, "errors.csv")
    elif category == "debug":
        file_path = os.path.join(logger_path, "debug.csv")
    else:
        raise ValueError(f"Unsupported logging category: {category}. Use 'error' or 'debug'.")

    append_row_to_csv(file_path, {
        "id_column": id_column,
        "json_text": json_text,
        "error": error,
        "request.text": request_text
    })