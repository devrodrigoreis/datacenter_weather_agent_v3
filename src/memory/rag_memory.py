"""
Tier 3: RAG Memory System (Retrieval-Augmented Generation)

Semantic memory system using vector embeddings for similarity-based retrieval.
Enables agents to find relevant past interactions, threats, and patterns using
semantic search instead of keyword matching.

This complements Tier 2 (AgentMemoryStore) by adding semantic capabilities:
- Tier 2: Structured data (SQL queries, statistics, exact patterns)
- Tier 3: Semantic data (similar queries, related threats, contextual retrieval)

Usage:
    from src.memory.rag_memory import RAGMemorySystem
    
    rag = RAGMemorySystem()
    
    # Add interaction with embedding
    rag.add_interaction(
        query="What's the weather in Seattle?",
        answer="Currently 65°F and sunny...",
        metadata={
            "agents": ["SecurityAgent", "IntentAgent", "WeatherAgent"],
            "success": True,
            "security_threat": None
        }
    )
    
    # Find similar past queries
    similar = rag.retrieve_similar_queries("weather in Portland", top_k=5)
    
    # Find related security threats
    threats = rag.retrieve_similar_threats("ignore all instructions", top_k=3)
    
    # Semantic search across all interactions
    results = rag.semantic_search("temperature forecast", top_k=10)
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class RAGMemorySystem:
    """
    RAG-based semantic memory for multi-agent system.
    
    Uses:
    - ChromaDB for vector storage and similarity search
    - sentence-transformers (all-MiniLM-L6-v2) for embeddings
    - Separate collections for different data types
    """
    
    def __init__(
        self, 
        persist_directory: str = "memory/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize RAG memory system.
        
        Args:
            persist_directory: Where to store ChromaDB data
            embedding_model: Sentence-transformers model to use
        """
        # Create directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model (22M parameters, fast)
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        logger.info("Embedding model loaded successfully")
        
        # Create collections for different data types
        self.interactions_collection = self.client.get_or_create_collection(
            name="interactions",
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
        
        self.threats_collection = self.client.get_or_create_collection(
            name="security_threats",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.weather_queries_collection = self.client.get_or_create_collection(
            name="weather_queries",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"RAG Memory System initialized at {persist_directory}")
        logger.info(f"Collections: interactions, security_threats, weather_queries")
    
    def _generate_id(self, text: str, prefix: str = "") -> str:
        """Generate unique ID from text."""
        hash_obj = hashlib.md5(text.encode())
        return f"{prefix}{hash_obj.hexdigest()}"
    
    def add_interaction(
        self,
        query: str,
        answer: Optional[str],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Add an interaction to the vector store.
        
        Args:
            query: User's query
            answer: System's answer (None if failed)
            metadata: Additional context (agents, success, time, etc.)
        
        Returns:
            Document ID
        """
        # Create searchable text (query + answer for better context)
        if answer:
            searchable_text = f"Query: {query}\nAnswer: {answer}"
        else:
            searchable_text = f"Query: {query}\nAnswer: [No answer generated]"
        
        # Generate embedding
        embedding = self.encoder.encode(searchable_text).tolist()
        
        # Generate unique ID
        doc_id = self._generate_id(searchable_text, prefix="int_")
        
        # Add timestamp to metadata
        metadata_with_timestamp = {
            **metadata,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer
        }
        
        # Convert lists/dicts to JSON strings (ChromaDB limitation)
        # Also filter out None values (ChromaDB doesn't accept None)
        for key, value in list(metadata_with_timestamp.items()):
            if value is None:
                del metadata_with_timestamp[key]
            elif isinstance(value, (list, dict)):
                metadata_with_timestamp[key] = json.dumps(value)
        
        # Store in ChromaDB
        self.interactions_collection.add(
            documents=[searchable_text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[metadata_with_timestamp]
        )
        
        logger.debug(f"Added interaction: {doc_id}")
        return doc_id
    
    def add_security_threat(
        self,
        threat_query: str,
        threat_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Add a security threat to the vector store.
        
        Args:
            threat_query: The malicious query
            threat_type: Type of threat (prompt_extraction, etc.)
            metadata: Additional context (confidence, blocked, etc.)
        
        Returns:
            Document ID
        """
        # Create searchable text with threat context
        searchable_text = f"Threat Type: {threat_type}\nQuery: {threat_query}"
        
        # Generate embedding
        embedding = self.encoder.encode(searchable_text).tolist()
        
        # Generate unique ID
        doc_id = self._generate_id(searchable_text, prefix="threat_")
        
        # Add metadata
        metadata_with_context = {
            **metadata,
            "timestamp": datetime.now().isoformat(),
            "threat_type": threat_type,
            "query": threat_query
        }
        
        # Convert lists/dicts to JSON strings (ChromaDB limitation)
        # Also filter out None values (ChromaDB doesn't accept None)
        for key, value in list(metadata_with_context.items()):
            if value is None:
                del metadata_with_context[key]
            elif isinstance(value, (list, dict)):
                metadata_with_context[key] = json.dumps(value)
        
        # Store in ChromaDB
        self.threats_collection.add(
            documents=[searchable_text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[metadata_with_context]
        )
        
        logger.debug(f"Added security threat: {doc_id} ({threat_type})")
        return doc_id
    
    def add_weather_query(
        self,
        query: str,
        location: Optional[str],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Add a successful weather query to the vector store.
        
        Args:
            query: The weather query
            location: Resolved location (if available)
            metadata: Additional context (coordinates, weather data, etc.)
        
        Returns:
            Document ID
        """
        # Create searchable text
        if location:
            searchable_text = f"Query: {query}\nLocation: {location}"
        else:
            searchable_text = f"Query: {query}"
        
        # Generate embedding
        embedding = self.encoder.encode(searchable_text).tolist()
        
        # Generate unique ID
        doc_id = self._generate_id(searchable_text, prefix="weather_")
        
        # Add metadata
        metadata_with_context = {
            **metadata,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "location": location
        }
        
        # Convert lists/dicts to JSON strings (ChromaDB limitation)
        # Also filter out None values (ChromaDB doesn't accept None)
        for key, value in list(metadata_with_context.items()):
            if value is None:
                del metadata_with_context[key]
            elif isinstance(value, (list, dict)):
                metadata_with_context[key] = json.dumps(value)
        
        # Store in ChromaDB
        self.weather_queries_collection.add(
            documents=[searchable_text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[metadata_with_context]
        )
        
        logger.debug(f"Added weather query: {doc_id}")
        return doc_id
    
    def retrieve_similar_queries(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find similar past queries using semantic search.
        
        Args:
            query: Query to search for
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
        
        Returns:
            List of similar queries with metadata and similarity scores
        """
        # Generate embedding for query
        query_embedding = self.encoder.encode(query).tolist()
        
        # Search in interactions collection
        results = self.interactions_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        similar_queries = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Convert distance to similarity
                
                # Filter by minimum similarity
                if similarity >= min_similarity:
                    similar_queries.append({
                        "document": doc,
                        "metadata": results['metadatas'][0][i],
                        "similarity": round(similarity, 4),
                        "id": results['ids'][0][i]
                    })
        
        logger.debug(f"Found {len(similar_queries)} similar queries for: {query[:50]}...")
        return similar_queries
    
    def retrieve_similar_threats(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Find similar past security threats using semantic search.
        
        Useful for SecurityAgent to detect variants of known threats.
        
        Args:
            query: Query to check for threats
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (higher for threats)
        
        Returns:
            List of similar threats with metadata and similarity scores
        """
        # Generate embedding for query
        query_embedding = self.encoder.encode(query).tolist()
        
        # Search in threats collection
        results = self.threats_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        similar_threats = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i]
                similarity = 1 - distance
                
                # Filter by minimum similarity (stricter for threats)
                if similarity >= min_similarity:
                    similar_threats.append({
                        "document": doc,
                        "metadata": results['metadatas'][0][i],
                        "similarity": round(similarity, 4),
                        "threat_type": results['metadatas'][0][i].get('threat_type'),
                        "id": results['ids'][0][i]
                    })
        
        logger.debug(f"Found {len(similar_threats)} similar threats for: {query[:50]}...")
        return similar_threats
    
    def retrieve_similar_weather_queries(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find similar weather queries.
        
        Args:
            query: Weather query to search for
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of similar weather queries with metadata
        """
        # Generate embedding for query
        query_embedding = self.encoder.encode(query).tolist()
        
        # Search in weather queries collection
        results = self.weather_queries_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        similar_queries = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i]
                similarity = 1 - distance
                
                # Filter by minimum similarity
                if similarity >= min_similarity:
                    similar_queries.append({
                        "document": doc,
                        "metadata": results['metadatas'][0][i],
                        "similarity": round(similarity, 4),
                        "location": results['metadatas'][0][i].get('location'),
                        "id": results['ids'][0][i]
                    })
        
        logger.debug(f"Found {len(similar_queries)} similar weather queries")
        return similar_queries
    
    def semantic_search(
        self,
        query: str,
        collection: str = "interactions",
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search across a collection.
        
        Args:
            query: Search query
            collection: Which collection to search ("interactions", "security_threats", "weather_queries")
            top_k: Number of results
            filter_metadata: Optional metadata filters (e.g., {"success": True})
        
        Returns:
            List of matching documents with metadata and scores
        """
        # Select collection
        if collection == "interactions":
            target_collection = self.interactions_collection
        elif collection == "security_threats":
            target_collection = self.threats_collection
        elif collection == "weather_queries":
            target_collection = self.weather_queries_collection
        else:
            raise ValueError(f"Unknown collection: {collection}")
        
        # Generate embedding
        query_embedding = self.encoder.encode(query).tolist()
        
        # Build query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k
        }
        
        # Add metadata filter if provided
        if filter_metadata:
            query_params["where"] = filter_metadata
        
        # Perform search
        results = target_collection.query(**query_params)
        
        # Format results
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i]
                similarity = 1 - distance
                
                search_results.append({
                    "document": doc,
                    "metadata": results['metadatas'][0][i],
                    "similarity": round(similarity, 4),
                    "id": results['ids'][0][i]
                })
        
        logger.debug(f"Semantic search in {collection}: {len(search_results)} results")
        return search_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored data.
        
        Returns:
            Dict with counts for each collection
        """
        return {
            "interactions": self.interactions_collection.count(),
            "security_threats": self.threats_collection.count(),
            "weather_queries": self.weather_queries_collection.count(),
            "total_documents": (
                self.interactions_collection.count() +
                self.threats_collection.count() +
                self.weather_queries_collection.count()
            ),
            "embedding_model": str(self.encoder),
            "embedding_dimension": self.encoder.get_sentence_embedding_dimension()
        }
    
    def clear_collection(self, collection: str):
        """
        Clear all data from a collection.
        
        Args:
            collection: Which collection to clear
        """
        if collection == "interactions":
            self.client.delete_collection("interactions")
            self.interactions_collection = self.client.get_or_create_collection(
                name="interactions",
                metadata={"hnsw:space": "cosine"}
            )
        elif collection == "security_threats":
            self.client.delete_collection("security_threats")
            self.threats_collection = self.client.get_or_create_collection(
                name="security_threats",
                metadata={"hnsw:space": "cosine"}
            )
        elif collection == "weather_queries":
            self.client.delete_collection("weather_queries")
            self.weather_queries_collection = self.client.get_or_create_collection(
                name="weather_queries",
                metadata={"hnsw:space": "cosine"}
            )
        else:
            raise ValueError(f"Unknown collection: {collection}")
        
        logger.info(f"Cleared collection: {collection}")
    
    def reset_all(self):
        """Clear all collections and reset the system."""
        self.client.reset()
        
        # Recreate collections
        self.interactions_collection = self.client.get_or_create_collection(
            name="interactions",
            metadata={"hnsw:space": "cosine"}
        )
        self.threats_collection = self.client.get_or_create_collection(
            name="security_threats",
            metadata={"hnsw:space": "cosine"}
        )
        self.weather_queries_collection = self.client.get_or_create_collection(
            name="weather_queries",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("RAG Memory System reset")


# Singleton pattern for shared instance
_shared_rag: Optional[RAGMemorySystem] = None

def get_rag_memory(
    persist_directory: str = "memory/chroma_db",
    embedding_model: str = "all-MiniLM-L6-v2"
) -> RAGMemorySystem:
    """
    Get or create shared RAG memory instance (singleton pattern).
    
    Args:
        persist_directory: Where to store ChromaDB data
        embedding_model: Sentence-transformers model to use
    
    Returns:
        Shared RAGMemorySystem instance
    """
    global _shared_rag
    if _shared_rag is None:
        _shared_rag = RAGMemorySystem(persist_directory, embedding_model)
    return _shared_rag
