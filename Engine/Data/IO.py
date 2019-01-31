import pandas as pd


class IO(object):
    def __init__(self):
        pass

    @classmethod
    def read_xlsx(cls, input_file_directory, input_sheet_name=None):
        """
        Read an excel file(.xlsx, not .xls).

        :param input_file_directory: file directory to read
        :param input_sheet_name: list of excel sheet if it has more than 2 sheets
        :return: pandas DataFrame
        """


        xls = pd.ExcelFile(input_file_directory)

        if not input_sheet_name:
            input_sheet_name = xls.sheet_names
        df = pd.DataFrame()

        for sheet_name in input_sheet_name:
            df_sheet = xls.parse(sheet_name)
            df = pd.concat([df, df_sheet])

        return df

    @classmethod
    def read_csv(cls, input_file_directory):
        """
        Read a csv file.

        :param input_file_directory: file directory to read
        :return: pandas DataFrame
        """
        df = pd.read_csv(input_file_directory, engine='python', encoding='utf-8')
        return df

    @classmethod
    def read_pickle(cls, input_file_directory):
        """
        Read a pickle file.

        :param input_file_directory: file directory to read
        :return: pandas DataFrame
        """
        df = pd.read_pickle(input_file_directory)
        return df

    @classmethod    
    def write_xlsx(cls, df, output_file_directory):
        """
        Write pandas DataFrame into an excel.

        :param df: pandas DataFrame
        :param output_file_directory: directory to save file
        :return: None
        """
        writer = pd.ExcelWriter(output_file_directory, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='output')
        writer.save()

    @classmethod
    def write_csv(cls, df, output_file_directory):
        """
        Write pandas DataFrame into a csv file.

        :param df: pandas DataFrame
        :param output_file_directory: directory to save file.
        :return: None
        """
        df.to_csv(output_file_directory, encoding='utf-8')

    @classmethod
    def write_pickle(cls, df, output_file_directory):
        """
        Write pandas DataFrame into a pickle file.

        :param df: pandas DataFrame
        :param output_file_directory: directory to save file.
        :return: None
        """
        df.to_pickle(output_file_directory)
