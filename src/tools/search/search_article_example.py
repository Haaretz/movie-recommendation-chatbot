from config.load_config import load_config
from src.chat.tools.search.search_article_core import SearchArticle

if __name__ == "__main__":
    query_text = "שלמה ארצי"

    config = load_config("config/config.yaml")
    search_article = SearchArticle(config)

    # relevant_docs_with_date_filter = search_article.retrieve_relevant_documents(
    #     query_text,
    #     brand=None,
    #     writer_name=['בן שלו'],
    #     publish_time_start=None,
    #     publish_time_end=None,
    #     primary_section=None,
    #     secondary_section=None,
    #     tags=None,
    #     article_type=None
    # )

    # print(relevant_docs_with_date_filter)

    # Example of using payload-only search
    payload_filtered_docs = search_article.retrieve_documents_by_payload(
        brand="הארץ",  #
        writer_name=None,
        publish_time_start=None,
        publish_time_end=None,
        primary_section=["חדשות"],
        secondary_section=None,
        tags=None,
        article_type=None,
    )
    print("\nDocuments retrieved by payload filter:")
    print(payload_filtered_docs)
