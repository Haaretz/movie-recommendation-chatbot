from typing import List

from qdrant_client import models

from constant import BRAND_MAPPING


class SearchArticleFilters:
    def __init__(self):
        pass

    def _create_qdrant_filter(
        self,
    ) -> models.Filter:
        """
        Create a Qdrant filter object based on fixed filter parameters
        to match the SQL query conditions.

        Filters for articles from 'HTZ' brand that are either:
        1. Written by 'ניב הדס' and in 'television', 'טלוויזיה', 'קולנוע', or 'cinema' secondary sections.
        2. Of type 'reviewArticle' or 'ReviewArticle'.

        Returns:
            Qdrant Filter object.
        """
        filter_conditions = []

        # 1. Brand filter (always 'HTZ')
        filter_conditions.append(
            models.FieldCondition(key="brand", match=models.MatchValue(value=BRAND_MAPPING.get('HTZ', 'HTZ')))
        )

        # 2. OR condition (should)
        or_conditions = []

        # 2.1. Condition A: 
        condition_a_filters = models.Filter(must=[
            models.FieldCondition(key="writer_name", match=models.MatchAny(any=['ניב הדס'])),
            models.FieldCondition(key="section_secondary", match=models.MatchAny(any=['television', 'טלוויזיה', 'קולנוע', 'cinema']))
        ])
        
        or_conditions.append(condition_a_filters) 

        # 2.2. Condition B: 
        condition_a_filters = models.Filter(must=[
            models.FieldCondition(key="article_type", match=models.MatchAny(any=['reviewArticle', 'ReviewArticle'])),
            models.FieldCondition(key="section_secondary", match=models.MatchAny(any=['television', 'טלוויזיה', 'קולנוע', 'cinema']))
        ])
        or_conditions.append(condition_a_filters)
        
        # 2.3. Condition C:
        condition_c_filters = models.Filter(must=[models.FieldCondition(key="writer_name", match=models.MatchAny(any=['חן חדד']))])
        or_conditions.append(condition_c_filters)


        # Combine all conditions with MUST and SHOULD
        final_filter = models.Filter(
            must=filter_conditions, 
            should=or_conditions, 
            must_not=[] 
        )

        return final_filter

