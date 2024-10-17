import re
import string
import sys
import tiktoken
import tomli

from openai import OpenAI

class GPTCommons:
    """
    A utility class for common operations with the GPT model.
    """
    def __init__(self, api_key, gptmodel, maxtokens, temperature, organization):
        """
        Initializes GPTCommons with predefined constants and sets the API key.

        Args:
        api_key (str): The API key for authenticating with the OpenAI API.
        gptmodel (str): The OpenAI model to use for tokenization.
        maxtokens (int): The maximum number of tokens allowed.
        temperature (float): The temperature to use for generating the completion.

        Constants:
        - SPECIAL_CHARACTERS (str): Punctuation and special characters used for text cleaning.
        - PATTERN (re.Pattern): Compiled regular expression pattern for cleaning text.
        """
        self.SPECIAL_CHARACTERS = string.punctuation + "“”‘’"
        self.PATTERN = re.compile(r'[\n\s]+')
        self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
        self.gptmodel = gptmodel
        self.maxtokens = maxtokens
        self.temperature = temperature
        self.organization = organization

    @staticmethod
    def initialize_gpt_commons(configfile):
        # Reading out OpenAI API keys and organization
        try:
            with open(configfile,"rb") as f:
                data = tomli.load(f)
        except Exception as e:
            print(f"Error: Unable to read openai.toml file.")
            print(e)
            sys.exit(1)

        try:
            api_key = data["openai"]["apikey"]
            if not api_key:
                raise ValueError("API key is missing or empty in the configuration.")
        except KeyError:
            raise KeyError("API key is mandatory and missing in the configuration.")

        try:
            gptmodel = data["openai"]["model"]
            if not gptmodel:
                raise ValueError("Model is missing or empty in the configuration.")
        except KeyError:
            raise KeyError("Model is missing in the configuration.")
        
        try:
            organization = data["openai"].get("organization", None)
        except KeyError:
            raise KeyError("Organization is missing in the configuration.")

        try:
            maxtokens = int(data["openai"]["maxtokens"])
        except KeyError:
            raise KeyError("Max tokens is mandatory and missing in the configuration.")
        except ValueError:
            raise ValueError("Max tokens must be an integer.")

        try:
            temperature = float(data["openai"]["temperature"])
            if not (0 <= temperature <= 1):
                raise ValueError("Temperature must be between 0 and 1.")
        except KeyError:
            raise KeyError("Temperature is mandatory and missing in the configuration.")
        except ValueError:
            raise ValueError("Temperature must be a float between 0 and 1.")

        # Initialize GPT utilities module
        commons = GPTCommons(api_key=api_key, gptmodel=gptmodel, maxtokens=maxtokens, temperature=temperature, organization=organization)
        return commons

    def get_api_key(self) -> str:
        return self.api_key

    def get_gptmodel(self) -> str:
        return self.gptmodel
    
    def get_organization(self) -> str:
        return self.organization

    def get_maxtokens(self) -> int:
        return self.maxtokens

    def get_temperature(self) -> float:
        return self.temperature

    def reduce_to_max_tokens(self, text) -> str:
        """
        Reduces the input text to a maximum number of tokens for the specified OpenAI model.

        Args:
        text (str): The input text to be reduced.
        max_tokens (int): The maximum number of tokens allowed.
        gptmodel (str): The OpenAI model to use for tokenization (default is "gpt-3.5-turbo").

        Returns:
        str: The reduced text.
        """
        if not isinstance(self.get_maxtokens(), int):
            raise ValueError("max_tokens must be an integer.")

        # Initialize the tokenizer for the specified model
        tokenizer = tiktoken.encoding_for_model(self.get_gptmodel())

        # Tokenize the input text
        tokens = tokenizer.encode(text)

        # Truncate the tokens to the maximum allowed number
        truncated_tokens = tokens[:self.get_maxtokens()]

        # Convert the tokens back to text
        reduced_text = tokenizer.decode(truncated_tokens)

        return reduced_text

    def clean_text(self, text) -> str:
        """
        Cleans a given text by replacing line breaks, consecutive whitespace, and handling special characters.

        Args:
        text (str): The input text to be cleaned.

        Returns:
        str: The cleaned text.

        Example:
        >>> dirty_text = "This is a\ndirty    text!!"
        >>> clean_text(dirty_text)
        'This is a dirty text  '
        """
        # Replace line breaks and consecutive whitespace with a single space
        text = re.sub(self.PATTERN, ' ', text).strip()

        # Handle special characters (replace with spaces or remove them)
        text = ''.join(char if char not in self.SPECIAL_CHARACTERS else ' ' for char in text)

        return text

    def get_chat_completion(self, prompt) -> str:
        """
        Retrieves a completion using the OpenAI ChatCompletion API with the specified model and parameters.

        Args:
        prompt (str): The user's input or prompt for generating the completion.
        model (str): The model to use for generating the completion.
        temperature (float): The temperature to use for generating the completion.

        Returns:
        str: The generated completion.
        """
        response = self.client.chat.completions.create(
                messages=[
                        {
                            "role": "system",
                            "content": prompt,
                         }
                        ],
                model=self.get_gptmodel(),
                temperature=self.get_temperature()
                )
        return response.choices[0].message.content.strip()
    
    def get_arg(self, arg_name, arg_descriptions, default=None) -> str:
        """
        Retrieves the value of a command-line argument by its name from the sys.argv list.

        Args:
        arg_name (str): The name of the command-line argument to retrieve.
        arg_descriptions (dict): A dictionary containing argument names as keys and their descriptions as values.
        default: The default value to return if the argument is not found (default is None).

        Returns:
        str or default: The value of the specified command-line argument or the default value if not found.

        If '--help' is present in the command-line arguments, it prints the usage message along with descriptions of all arguments and exits the program.

        Example:
        >>> # Assuming the command-line arguments are ['--lang', 'English', '--url', 'example.com']
        >>> get_arg('--lang', {'--lang': 'Language (default: English)', '--url': 'PDF URL'}, 'Spanish')
        'English'
        >>> get_arg('--url', {'--lang': 'Language (default: English)', '--url': 'PDF URL'}, 'localhost')
        'example.com'
        >>> get_arg('--port', {'--lang': 'Language (default: English)', '--url': 'PDF URL'}, 8080)
        8080
        """
        if "--help" in sys.argv:
            script_name = sys.argv[0]
            print(f"Usage: python {script_name} [options]")
            print("Options:")
            for arg, desc in arg_descriptions.items():
                print(f"\t{arg}\t\t{desc}")
            sys.exit(0)

        try:
            arg_index = sys.argv.index(arg_name)
            arg_value = sys.argv[arg_index + 1]
            return arg_value
        except (IndexError, ValueError):
            return default

    def split_into_chunks(self, text, chunk_size=1000, overlap_percentage=1) -> list[str]:
        """
        Splits a given text into smaller chunks with a specified chunk size and overlap percentage.

        Args:
        text (str): The input text to be split into chunks.
        chunk_size (int, optional): The desired size of each chunk (default is 1000).
        overlap_percentage (float, optional): The percentage of overlap between consecutive chunks (default is 1).
                                            A value of 0 means no overlap, and 1 means 100% overlap.

        Returns:
        list of str: A list containing the text chunks.

        Example:
        >>> text = "This is an example text that needs to be split into smaller chunks."
        >>> split_into_chunks(text, chunk_size=20, overlap_percentage=0.5)
        ['This is an example ', ' example text that ne', 'text that needs to be', ' to be split into smal', ' smaller chunks.']
        """
        #split the web content into chunks of 1000 characters
        text = self.clean_text( text)

        # Calculate the number of overlapping characters
        overlap_chars = int(chunk_size * overlap_percentage)

        # Initialize a list to store the chunks
        chunks = []

        # Loop through the text with the overlap
        for i in range(0, len(text), chunk_size - overlap_chars):
            # Determine the end index of the current chunk
            end_idx = i + chunk_size

            # Slice the text to form a chunk
            chunk = text[i:end_idx]

            # Append the chunk to the list
            chunks.append(chunk)

        return chunks