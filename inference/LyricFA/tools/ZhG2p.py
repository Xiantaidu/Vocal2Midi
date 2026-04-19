tone_to_number = {
    'ā': ('a', '1'), 'á': ('a', '2'), 'ǎ': ('a', '3'), 'à': ('a', '4'),
    'ō': ('o', '1'), 'ó': ('o', '2'), 'ǒ': ('o', '3'), 'ò': ('o', '4'),
    'ē': ('e', '1'), 'é': ('e', '2'), 'ě': ('e', '3'), 'è': ('e', '4'),
    'ī': ('i', '1'), 'í': ('i', '2'), 'ǐ': ('i', '3'), 'ì': ('i', '4'),
    'ū': ('u', '1'), 'ú': ('u', '2'), 'ǔ': ('u', '3'), 'ù': ('u', '4'),
    'ḿ': ('m', '2'), 'ǹ': ('n', '4'),
    'ǖ': ('v', '1'), 'ǘ': ('v', '2'), 'ǚ': ('v', '3'), 'ǜ': ('v', '4'),
    'ü': ('v', '')
}


def tone_to_normal(pinyin, v_to_u=False):
    result = []
    for character in pinyin:
        if 'a' <= character <= 'z':
            result.append(character)
        else:
            result.append(tone_to_number.get(character, (character,))[0])
    result_str = ''.join(result)
    if v_to_u:
        result_str = result_str.replace('v', 'ü')
    return result_str


def is_letter(character):
    return ('a' <= character <= 'z') or ('A' <= character <= 'Z')


def is_special_letter(character):
    special_letter = "'-’"
    return character in special_letter


def is_hanzi(character):
    return 0x4e00 <= ord(character) <= 0x9fa5


def is_kana(character):
    return (0x3040 <= ord(character) <= 0x309F) or (0x30A0 <= ord(character) <= 0x30FF)


def is_special_kana(character):
    special_kana = "ャュョゃゅょァィゥェォぁぃぅぇぉ"
    return character in special_kana


def split_string(input_str):
    result = []
    position = 0
    while position < len(input_str):
        current_char = input_str[position]
        if is_letter(current_char) or is_special_letter(current_char):
            start = position
            while position < len(input_str) and (
                    is_letter(input_str[position]) or is_special_letter(input_str[position])):
                position += 1
            result.append(input_str[start:position])
        elif is_hanzi(current_char) or current_char.isdigit():
            result.append(input_str[position])
            position += 1
        elif is_kana(current_char):
            length = 2 if position + 1 < len(input_str) and is_special_kana(input_str[position + 1]) else 1
            result.append(input_str[position:position + length])
            position += length
        else:
            position += 1
    return result


import os

class ZhG2p:
    def __init__(self, language):
        self.phrases_map = {}
        self.trans_dict = {}
        self.word_dict = {}
        self.phrases_dict = {}
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if language == "mandarin":
            dict_directory = os.path.join(base_dir, "Dicts/mandarin")
        else:
            dict_directory = os.path.join(base_dir, "Dicts/cantonese")

        self.load_dict(dict_directory, "phrases_map.txt", self.phrases_map)
        self.load_dict_list(dict_directory, "phrases_dict.txt", self.phrases_dict)
        self.load_dict_list(dict_directory, "user_dict.txt", self.phrases_dict, " ")
        self.load_dict_list(dict_directory, "word.txt", self.word_dict)
        self.load_dict(dict_directory, "trans_word.txt", self.trans_dict)

    @staticmethod
    def load_dict(directory, file_name, result_map):
        dict_path = directory + "/" + file_name
        with open(dict_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    key, value = line.split(':')
                    result_map[key] = value

    @staticmethod
    def load_dict_list(directory, file_name, result_map, separator=','):
        dict_path = directory + "/" + file_name
        with open(dict_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    key, value = line.split(':', 1)
                    result_map[key] = value.split(separator)

    number_map = {
        "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
        "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"
    }

    @staticmethod
    def reset_zh(input_list, result, positions):
        final_result = input_list.copy()
        for index, position in enumerate(positions):
            final_result[position] = result[index]
        return " ".join(final_result)

    @staticmethod
    def remove_elements(list_to_modify, start_index, count):
        if 0 <= start_index < len(list_to_modify) and count > 0:
            count_to_remove = min(count, len(list_to_modify) - start_index)
            del list_to_modify[start_index:start_index + count_to_remove]

    @staticmethod
    def split_string_no_regex(input_str):
        result = []
        position = 0
        while position < len(input_str):
            current_char = input_str[position]
            if is_letter(current_char):
                start = position
                while position < len(input_str) and is_letter(input_str[position]):
                    position += 1
                result.append(input_str[start:position])
            elif is_hanzi(current_char) or current_char.isdigit():
                result.append(input_str[position])
                position += 1
            elif is_kana(current_char):
                length = 2 if position + 1 < len(input_str) and is_special_kana(input_str[position + 1]) else 1
                result.append(input_str[position:position + length])
                position += length
            else:
                position += 1
        return result

    def convert(self, input_text, include_tone=False, convert_number=False):
        return self.convert_list(self.split_string_no_regex(input_text), include_tone, convert_number)

    def zh_position(self, input_list, result, positions, convert_number):
        for index, value in enumerate(input_list):
            if value in self.word_dict or value in self.trans_dict:
                result.append(value)
                positions.append(index)
            elif convert_number and value in self.number_map:
                result.append(self.number_map[value])
                positions.append(index)

    def convert_list(self, input_list, include_tone=False, convert_number=False):
        processed_input = []
        input_positions = []
        self.zh_position(input_list, processed_input, input_positions, convert_number)
        clean_input = ''.join(processed_input)
        result = []
        cursor = 0

        while cursor < len(processed_input):
            raw_current_char = processed_input[cursor]
            current_char = self.traditional_to_simplified(raw_current_char)

            if current_char not in self.word_dict:
                result.append(current_char)
                cursor += 1
                continue

            if not self.is_polyphonic(current_char):
                result.append(self.get_default_pinyin(current_char))
                cursor += 1
            else:
                found = False
                for length in range(4, 1, -1):
                    if cursor + length <= len(processed_input):
                        sub_phrase = clean_input[cursor:cursor + length]
                        if sub_phrase in self.phrases_dict:
                            result.extend(self.phrases_dict[sub_phrase])
                            cursor += length
                            found = True

                        if cursor >= 1 and not found:
                            sub_phrase_1 = clean_input[cursor - 1:cursor + length - 1]
                            if sub_phrase_1 in self.phrases_dict:
                                result = result[:-1]
                                result.extend(self.phrases_dict[sub_phrase_1])
                                cursor += length - 1
                                found = True

                    if 0 <= cursor + 1 - length < cursor + 1 and not found and cursor < len(processed_input):
                        x_sub_phrase = clean_input[cursor + 1 - length:cursor + 1]
                        if x_sub_phrase in self.phrases_dict:
                            self.remove_elements(result, cursor + 1 - length, length - 1)
                            result.extend(self.phrases_dict[x_sub_phrase])
                            cursor += 1
                            found = True

                    if 0 <= cursor + 2 - length < cursor + 2 and not found and cursor < len(processed_input):
                        x_sub_phrase = clean_input[cursor + 2 - length:cursor + 2]
                        if x_sub_phrase in self.phrases_dict:
                            self.remove_elements(result, cursor + 2 - length, length - 1)
                            result.extend(self.phrases_dict[x_sub_phrase])
                            cursor += 2
                            found = True

                if not found:
                    result.append(self.get_default_pinyin(current_char))
                    cursor += 1

        if not include_tone:
            result = [x[:-1] if x[-1].isdigit() else x for x in result]

        result = [tone_to_normal(x) for x in result]
        return self.reset_zh(input_list, result, input_positions)

    def is_polyphonic(self, text):
        return text in self.phrases_map

    def traditional_to_simplified(self, text):
        return self.trans_dict[text] if text in self.trans_dict else text

    def get_default_pinyin(self, text):
        return self.word_dict[self.traditional_to_simplified(text)][0] if text in self.word_dict else None

    def get_all_pinyin(self, text):
        return self.word_dict[self.traditional_to_simplified(text)] if text in self.word_dict else None
