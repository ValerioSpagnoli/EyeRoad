import torch

class AttnConverter(object):
    def __init__(self, character, batch_max_length, go_last=False):
        """
        Initialize the AttnConverter object.

        Args:
            character (str): The set of characters to be used in encoding and decoding.
            batch_max_length (int): Maximum length of each batch text.
            go_last (bool): Whether to add special tokens at the end of characters.
        """
        self.character = list(character)
        self.batch_max_length = batch_max_length + 1

        if go_last:
            self.character += ['[s]', '[GO]']
        else:
            self.character += ['[GO]', '[s]']

        # Create dictionary of key (char) values (idx)
        self.dict = {value: index for index, value in enumerate(self.character)}
        self.ignore_id = self.dict['[GO]']


    def train_encode(self, text):
        length = [len(s) + 1 for s in text]
        batch_text = torch.full((len(text), self.batch_max_length + 1), self.ignore_id, dtype=torch.long)
        
        for idx, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[idx][1:1 + len(text)] = torch.LongTensor(text)
        batch_text_input = batch_text[:, :-1]
        batch_text_target = batch_text[:, 1:]

        return batch_text_input, torch.IntTensor(length), batch_text_target

    def test_encode(self, text):
        if isinstance(text, (list, tuple)):
            num = len(text)
        elif isinstance(text, int):
            num = text
        else:
            raise TypeError(f'Type of text should in (list, tuple, int) but got {type(text)}')
        batch_text = torch.LongTensor(num, 1).fill_(self.ignore_id)
        length = [1 for i in range(num)]

        return batch_text, torch.IntTensor(length), batch_text

    def decode(self, text_index):
        texts = []
        batch_size = text_index.shape[0]
        for index in range(batch_size):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            text = text[:text.find('[s]')]
            texts.append(text)

        return texts
