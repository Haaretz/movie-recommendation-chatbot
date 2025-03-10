import re


def get_haaretz_or_themarker_urls_from_string(text):
    """
    Extracts and returns a list of URLs from a string that are from 'Haaretz' or 'TheMarker' websites.
    This is a shorter version using a more direct regex approach.

    Args:
        text: string, the input string to search for URLs within.

    Returns:
        list: A list of URLs from 'Haaretz' or 'TheMarker' found in the string.
              Returns an empty list if no matching URLs are found.
    """
    full_url_regex = r"(https?://(?:www\.)?(?:haaretz\.co\.il|themarker\.com)/[^\s]*)"
    return re.findall(full_url_regex, text)  # Directly find all matching URLs


if __name__ == "__main__":
    # Example usage (same as before):
    text_with_urls = """
    Here are some articles:
    - https://www.haaretz.co.il/digital/podcast/weekly/2025-02-18/ty-article-podcast/00000195-1854-d8ed-abf7-faff375a0000
    - Check out this one too: https://www.themarker.com/markets/2025-02-18/ty-article-live/00000195-17e3-d4b6-abfd-5fe3bda50000 and also maybe http://www.haaretz.co.il/news/world.
    - Not related: https://www.ynet.co.il/news/economic and https://www.google.com.
    - Another Haaretz link: https://haaretz.co.il/news/economy.
    - And a TheMarker blog: https://themarker.com/blogs/tech.
    """

    found_urls = get_haaretz_or_themarker_urls_from_string(text_with_urls)

    if found_urls:
        print("Found Haaretz or TheMarker URLs:")
        for url in found_urls:
            print(f"- {url}")
    else:
        print("No Haaretz or TheMarker URLs found in the text.")

    text_with_urls = "אין פה לינק"
    found_urls = get_haaretz_or_themarker_urls_from_string(text_with_urls)
    print(found_urls)  # Output: []
