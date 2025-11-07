"""
Web search and citation tool for the Council of Infinite Innovators.
"""

import os
import httpx
from typing import List, Dict, Optional

async def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Perform web search using Tavily API (or fallback to basic search).
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, url, and snippet
    """
    api_key = os.getenv("TAVILY_API_KEY")
    
    if not api_key:
        # Return mock results if no API key
        return [
            {
                "title": f"Search result for: {query}",
                "url": "https://example.com",
                "snippet": "Web search functionality requires TAVILY_API_KEY to be configured."
            }
        ]
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": False,
                    "include_images": False,
                    "include_raw_content": False,
                    "max_results": max_results
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get("results", []):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", "")[:200] + "..."
                    })
                
                return results
            else:
                return [
                    {
                        "title": "Search Error",
                        "url": "",
                        "snippet": f"Search API returned status code: {response.status_code}"
                    }
                ]
                
    except Exception as e:
        return [
            {
                "title": "Search Error", 
                "url": "",
                "snippet": f"Search failed: {str(e)}"
            }
        ]

async def get_web_citations(query: str, context: str) -> List[str]:
    """
    Get relevant citations for a given query and context.
    
    Args:
        query: Original query
        context: Context needing citations
        
    Returns:
        List of citation URLs
    """
    search_results = await web_search(query, max_results=3)
    return [result["url"] for result in search_results if result["url"]]

def format_search_results(results: List[Dict[str, str]]) -> str:
    """Format search results for inclusion in agent responses."""
    if not results:
        return "No search results available."
    
    formatted = "🔍 **Web Search Results:**\n\n"
    for i, result in enumerate(results, 1):
        formatted += f"{i}. **{result['title']}**\n"
        formatted += f"   {result['snippet']}\n"
        formatted += f"   🔗 {result['url']}\n\n"
    
    return formatted