# import re

# import pandas as pd
# from Levenshtein import distance

# WRITER_NAME = pd.read_csv(r"data/writer_names.csv")
# WRITER_NAME = WRITER_NAME[WRITER_NAME["unique_article_count"] > 30]["writer_name"].tolist()

# SECTION_PRIMARY = pd.read_csv(r"data/section_primary.csv")["section_primary"].tolist()
# SECTION_SECONDARY = pd.read_csv(r"data/section_secondary.csv")["section_secondary"].tolist()


# def filter_writer_name(prompt):
#     if re.search(r"כתב(ת|ים)?", prompt) is None:
#         return None
#     writer_name_variations = {}

#     for full_name in WRITER_NAME:
#         name_parts = full_name.split()
#         if len(name_parts) >= 2:
#             first_name = name_parts[0]
#             last_name = " ".join(name_parts[1:])
#             variations = [full_name, f"{last_name} {first_name}"]
#         else:
#             variations = [full_name]

#         writer_name_variations[full_name] = variations

#     for original_name, variations in writer_name_variations.items():
#         for variation in variations:
#             if variation in prompt:
#                 return original_name

#     distances = []
#     for original_name, variations in writer_name_variations.items():
#         min_distance = float("inf")  # Initialize with infinity for finding the minimum
#         for variation in variations:
#             current_distance = distance(prompt, variation)
#             min_distance = min(min_distance, current_distance)  # Track the minimum distance across variations
#         distances.append((min_distance, original_name))  # Store min distance with original name

#     distances.sort(key=lambda x: x[0])  # Sort by minimum Levenshtein distance

#     closest_writers = [writer for dist, writer in distances[:4]]  # Get top 4 closest original names

#     if closest_writers:
#         return closest_writers
#     return None


# def filter_section_primary(prompt):
#     if "קטגוריה" not in prompt:
#         return None
#     for section in SECTION_PRIMARY:
#         if section in prompt:
#             return section

#     distances = []
#     for section in SECTION_PRIMARY:
#         distances.append((distance(prompt, section), section))

#     distances.sort(key=lambda x: x[0])
#     closest_sections = [section for dist, section in distances[:4]]
#     if closest_sections:
#         return closest_sections
#     return None


# def filter_section_secondary(prompt):
#     if "תת קטגוריה" not in prompt:
#         return None
#     for section in SECTION_SECONDARY:
#         if section in prompt:
#             return section

#     distances = []
#     for section in SECTION_SECONDARY:
#         distances.append((distance(prompt, section), section))

#     distances.sort(key=lambda x: x[0])
#     closest_sections = [section for dist, section in distances[:4]]
#     if closest_sections:
#         return closest_sections
#     return None


# if __name__ == "__main__":
#     prompts_and_expected_outputs = [
#         # ("על איזה דיסקים חדשים הכתב שלו בן המליץ?", "בן שלו"),  # Typo, reversed order - should return closest
#         # ("על איזה דיסקים חדשים הכתב בן שלו המליץ?", "בן שלו"),  # Correct order - exact match
#         # ("על איזה דיסקים חדשים הכתב שלו בן המליץ?", "בן שלו"), # Reversed order with typo - should return closest
#         # ("על איזה דיסקים חדשים הכתב שלו בן?", "בן שלו"), # Reversed order, partial name - should return closest
#         # ("כתב בשם שלו בן המליץ על...", "בן שלו"), # Reversed order, name as subject - should return closest
#         # ("מאמר מאת בן שלו על מוזיקה", "בן שלו"), # Correct order in different context - exact match
#         # ("כתבה מאת שלו בן בנושא...", "בן שלו"), # Reversed order in different context - should return closest
#         # ("מי הכתב בן שלו?", "בן שלו"), # Question format, correct order - exact match
#         # ("מי הכתב שלו בן?", "בן שלו"), # Question format, reversed order - should return closest
#         ("אין שם כתב מתאים בטקסט", None),  # No writer name at all - None
#         # ("קטגוריה: מוזיקה", None), # Just category, no writer - None
#     ]

#     for prompt, expected_output in prompts_and_expected_outputs:
#         actual_output = filter_writer_name(prompt)
#         test_passed = False

#         if expected_output is None:
#             test_passed = actual_output is None
#         elif isinstance(expected_output, str):
#             test_passed = actual_output == expected_output
#         elif isinstance(
#             expected_output, list
#         ):  # In case we expect a list of close matches (not used in current expected outputs but for future flexibility)
#             test_passed = (
#                 isinstance(actual_output, list)
#                 and all(item in actual_output for item in expected_output)
#                 and len(actual_output) == len(expected_output)
#             )

#         if not test_passed:
#             print(f"Test Failed for Prompt: '{prompt}'")
#             print(f"Expected Output: {expected_output}")
#             print(f"Actual Output: {actual_output}")
#             print("-" * 30)
