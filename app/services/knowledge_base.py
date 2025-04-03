"""
ProjectPilot - AI-powered project management system
Knowledge Base system for storing and retrieving institutional knowledge.
"""

import os
import json
import logging
import time
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Union, Set, Tuple
import threading
from datetime import datetime
import sqlite3
from pathlib import Path
import re
from flask import current_app

logger = logging.getLogger(__name__)

class KnowledgeEntry:
    """A single entry in the knowledge base."""
    
    def __init__(self, 
                 title: str, 
                 content: str, 
                 category: str, 
                 tags: List[str] = None,
                 source: str = None,
                 entry_id: Optional[int] = None,
                 timestamp: Optional[datetime] = None,
                 author_id: Optional[int] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a knowledge entry.
        
        Args:
            title: Entry title
            content: Entry content
            category: Entry category
            tags: List of tags for easy searching
            source: Source of the knowledge
            entry_id: Entry ID (auto-generated if None)
            timestamp: Entry creation timestamp (now if None)
            author_id: ID of the user or bot who created the entry
            metadata: Additional metadata for the entry
        """
        self.title = title
        self.content = content
        self.category = category
        self.tags = tags or []
        self.source = source
        self.entry_id = entry_id
        self.timestamp = timestamp or datetime.now()
        self.author_id = author_id
        self.metadata = metadata or {}
        
        # Generate content hash for deduplication
        self.content_hash = self._generate_content_hash()
    
    def _generate_content_hash(self) -> str:
        """
        Generate a hash of the entry content for deduplication.
        
        Returns:
            Content hash
        """
        content_string = f"{self.title.lower()}|{self.content.lower()}"
        return hashlib.md5(content_string.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entry to dictionary.
        
        Returns:
            Dictionary representation of the entry
        """
        return {
            'id': self.entry_id,
            'title': self.title,
            'content': self.content,
            'category': self.category,
            'tags': self.tags,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'author_id': self.author_id,
            'content_hash': self.content_hash,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntry':
        """
        Create an entry from a dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            KnowledgeEntry instance
        """
        timestamp = datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp']
        
        return cls(
            title=data['title'],
            content=data['content'],
            category=data['category'],
            tags=data['tags'],
            source=data['source'],
            entry_id=data['id'],
            timestamp=timestamp,
            author_id=data['author_id'],
            metadata=data['metadata']
        )

class KnowledgeBase:
    """
    Knowledge Base system for storing and retrieving institutional knowledge
    and learnings throughout projects.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the knowledge base.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'knowledge.db')
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Vector cache for semantic search
        self.vector_cache = {}
        self.vector_cache_lock = threading.RLock()
        
        # Initialize database
        self._initialize_database()
        
        # Initialize embedding model (lazy loaded)
        self.embedding_model = None
        
        logger.info(f"Knowledge Base initialized with database at {self.db_path}")
    
    def add_entry(self, entry: KnowledgeEntry) -> int:
        """
        Add a knowledge entry to the knowledge base.
        
        Args:
            entry: Knowledge entry to add
            
        Returns:
            Entry ID
        """
        try:
            # Check for duplicates
            duplicate = self._check_duplicate(entry)
            if duplicate:
                logger.info(f"Duplicate entry detected: {entry.title}")
                return duplicate
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert entry
            cursor.execute('''
                INSERT INTO knowledge_entries (
                    title, content, category, tags, source, timestamp,
                    author_id, content_hash, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.title,
                entry.content,
                entry.category,
                json.dumps(entry.tags),
                entry.source,
                entry.timestamp.isoformat(),
                entry.author_id,
                entry.content_hash,
                json.dumps(entry.metadata)
            ))
            
            # Get the inserted ID
            entry_id = cursor.lastrowid
            
            # Update the entry ID
            entry.entry_id = entry_id
            
            # Add to search index
            self._update_search_index(entry)
            
            # Add to vector cache
            self._update_vector_cache(entry)
            
            # Commit and close
            conn.commit()
            conn.close()
            
            logger.info(f"Added knowledge entry: {entry.title} (ID: {entry_id})")
            
            return entry_id
            
        except Exception as e:
            logger.error(f"Error adding knowledge entry: {str(e)}")
            return -1
    
    def update_entry(self, entry_id: int, updates: Dict[str, Any]) -> bool:
        """
        Update a knowledge entry.
        
        Args:
            entry_id: Entry ID
            updates: Dictionary of field updates
            
        Returns:
            True if update was successful
        """
        try:
            # Get current entry
            current_entry = self.get_entry(entry_id)
            if not current_entry:
                logger.error(f"Entry not found: {entry_id}")
                return False
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build update query
            update_fields = []
            update_values = []
            
            if 'title' in updates:
                update_fields.append('title = ?')
                update_values.append(updates['title'])
                current_entry.title = updates['title']
            
            if 'content' in updates:
                update_fields.append('content = ?')
                update_values.append(updates['content'])
                current_entry.content = updates['content']
                
                # Update content hash
                content_hash = current_entry._generate_content_hash()
                update_fields.append('content_hash = ?')
                update_values.append(content_hash)
                current_entry.content_hash = content_hash
            
            if 'category' in updates:
                update_fields.append('category = ?')
                update_values.append(updates['category'])
                current_entry.category = updates['category']
            
            if 'tags' in updates:
                update_fields.append('tags = ?')
                update_values.append(json.dumps(updates['tags']))
                current_entry.tags = updates['tags']
            
            if 'source' in updates:
                update_fields.append('source = ?')
                update_values.append(updates['source'])
                current_entry.source = updates['source']
            
            if 'metadata' in updates:
                # Merge existing metadata with updates
                merged_metadata = current_entry.metadata.copy()
                merged_metadata.update(updates['metadata'])
                
                update_fields.append('metadata = ?')
                update_values.append(json.dumps(merged_metadata))
                current_entry.metadata = merged_metadata
            
            # Add entry_id to values
            update_values.append(entry_id)
            
            # Execute update
            cursor.execute(f'''
                UPDATE knowledge_entries
                SET {', '.join(update_fields)}
                WHERE id = ?
            ''', update_values)
            
            # Update search index
            self._update_search_index(current_entry)
            
            # Update vector cache
            self._update_vector_cache(current_entry)
            
            # Commit and close
            conn.commit()
            conn.close()
            
            logger.info(f"Updated knowledge entry: {current_entry.title} (ID: {entry_id})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating knowledge entry {entry_id}: {str(e)}")
            return False
    
    def delete_entry(self, entry_id: int) -> bool:
        """
        Delete a knowledge entry.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            True if deletion was successful
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete entry
            cursor.execute('DELETE FROM knowledge_entries WHERE id = ?', (entry_id,))
            
            # Delete from search index
            cursor.execute('DELETE FROM search_index WHERE entry_id = ?', (entry_id,))
            
            # Remove from vector cache
            with self.vector_cache_lock:
                if entry_id in self.vector_cache:
                    del self.vector_cache[entry_id]
            
            # Commit and close
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted knowledge entry: {entry_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting knowledge entry {entry_id}: {str(e)}")
            return False
    
    def get_entry(self, entry_id: int) -> Optional[KnowledgeEntry]:
        """
        Get a knowledge entry by ID.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            KnowledgeEntry or None if not found
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get entry
            cursor.execute('''
                SELECT id, title, content, category, tags, source, timestamp,
                       author_id, content_hash, metadata
                FROM knowledge_entries
                WHERE id = ?
            ''', (entry_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            # Create entry object
            entry = KnowledgeEntry(
                title=row[1],
                content=row[2],
                category=row[3],
                tags=json.loads(row[4]),
                source=row[5],
                entry_id=row[0],
                timestamp=datetime.fromisoformat(row[6]),
                author_id=row[7],
                metadata=json.loads(row[9])
            )
            
            # Override content hash with stored value
            entry.content_hash = row[8]
            
            return entry
            
        except Exception as e:
            logger.error(f"Error getting knowledge entry {entry_id}: {str(e)}")
            return None
    
    def search(self, 
              query: str, 
              category: Optional[str] = None, 
              tags: Optional[List[str]] = None,
              limit: int = 10,
              offset: int = 0) -> List[KnowledgeEntry]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            category: Filter by category
            tags: Filter by tags
            limit: Maximum number of results to return
            offset: Offset for pagination
            
        Returns:
            List of matching knowledge entries
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query
            search_terms = self._tokenize_query(query)
            
            # Search for each term in the search index
            query_parts = []
            query_params = []
            
            for term in search_terms:
                query_parts.append('SELECT entry_id FROM search_index WHERE term LIKE ?')
                query_params.append(f"%{term}%")
            
            entry_query = f'''
                SELECT id, title, content, category, tags, source, timestamp,
                       author_id, content_hash, metadata
                FROM knowledge_entries
                WHERE id IN (
                    {' INTERSECT '.join(query_parts)}
                )
            '''
            
            # Add category filter if provided
            if category:
                entry_query += ' AND category = ?'
                query_params.append(category)
            
            # Add tags filter if provided
            if tags:
                for tag in tags:
                    entry_query += ' AND tags LIKE ?'
                    query_params.append(f'%"{tag}"%')
            
            # Add limit and offset
            entry_query += ' LIMIT ? OFFSET ?'
            query_params.extend([limit, offset])
            
            # Execute query
            cursor.execute(entry_query, query_params)
            
            # Fetch results
            results = []
            for row in cursor.fetchall():
                entry = KnowledgeEntry(
                    title=row[1],
                    content=row[2],
                    category=row[3],
                    tags=json.loads(row[4]),
                    source=row[5],
                    entry_id=row[0],
                    timestamp=datetime.fromisoformat(row[6]),
                    author_id=row[7],
                    metadata=json.loads(row[9])
                )
                
                # Override content hash with stored value
                entry.content_hash = row[8]
                
                results.append(entry)
            
            conn.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return []
    
    def semantic_search(self, 
                      query: str,
                      limit: int = 5) -> List[Tuple[KnowledgeEntry, float]]:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of tuples (entry, similarity_score)
        """
        try:
            # Ensure embedding model is loaded
            if not self._ensure_embedding_model():
                logger.warning("Embedding model not available, falling back to regular search")
                return [(e, 1.0) for e in self.search(query, limit=limit)]
            
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            # Get all entries with embeddings
            entries_with_vectors = []
            
            with self.vector_cache_lock:
                # Use cached vectors
                for entry_id, vector in self.vector_cache.items():
                    entry = self.get_entry(entry_id)
                    if entry:
                        entries_with_vectors.append((entry, vector))
            
            # Calculate similarities
            results_with_scores = []
            
            for entry, vector in entries_with_vectors:
                similarity = self._calculate_similarity(query_embedding, vector)
                results_with_scores.append((entry, similarity))
            
            # Sort by similarity (descending)
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top results
            return results_with_scores[:limit]
            
        except Exception as e:
            logger.error(f"Error performing semantic search: {str(e)}")
            return []
    
    def get_categories(self) -> List[str]:
        """
        Get all categories in the knowledge base.
        
        Returns:
            List of categories
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get categories
            cursor.execute('SELECT DISTINCT category FROM knowledge_entries')
            
            # Fetch results
            categories = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            return categories
            
        except Exception as e:
            logger.error(f"Error getting categories: {str(e)}")
            return []
    
    def get_tags(self) -> List[str]:
        """
        Get all tags in the knowledge base.
        
        Returns:
            List of tags
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all tag arrays
            cursor.execute('SELECT tags FROM knowledge_entries')
            
            # Fetch results
            tag_sets = [json.loads(row[0]) for row in cursor.fetchall()]
            
            conn.close()
            
            # Flatten and deduplicate
            all_tags = []
            for tag_set in tag_sets:
                all_tags.extend(tag_set)
            
            return list(set(all_tags))
            
        except Exception as e:
            logger.error(f"Error getting tags: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with statistics
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total entries
            cursor.execute('SELECT COUNT(*) FROM knowledge_entries')
            total_entries = cursor.fetchone()[0]
            
            # Get entries by category
            cursor.execute('''
                SELECT category, COUNT(*) 
                FROM knowledge_entries 
                GROUP BY category
            ''')
            entries_by_category = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get entry counts by date
            cursor.execute('''
                SELECT DATE(timestamp), COUNT(*) 
                FROM knowledge_entries 
                GROUP BY DATE(timestamp)
                ORDER BY DATE(timestamp)
            ''')
            entries_by_date = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get authors with most entries
            cursor.execute('''
                SELECT author_id, COUNT(*) 
                FROM knowledge_entries 
                GROUP BY author_id
                ORDER BY COUNT(*) DESC
                LIMIT 5
            ''')
            top_authors = {row[0]: row[1] for row in cursor.fetchall() if row[0] is not None}
            
            conn.close()
            
            # Assemble statistics
            statistics = {
                "total_entries": total_entries,
                "categories": entries_by_category,
                "tags": len(self.get_tags()),
                "entries_by_date": entries_by_date,
                "top_authors": top_authors,
                "vector_cache_size": len(self.vector_cache)
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error getting knowledge base statistics: {str(e)}")
            return {"error": str(e)}
    
    def export_entries(self, 
                      category: Optional[str] = None, 
                      tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Export knowledge entries to a structured format.
        
        Args:
            category: Filter by category
            tags: Filter by tags
            
        Returns:
            Dictionary with exported entries
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query
            query = '''
                SELECT id, title, content, category, tags, source, timestamp,
                      author_id, content_hash, metadata
                FROM knowledge_entries
            '''
            
            params = []
            where_clauses = []
            
            # Add category filter if provided
            if category:
                where_clauses.append('category = ?')
                params.append(category)
            
            # Add tags filter if provided
            if tags:
                for tag in tags:
                    where_clauses.append('tags LIKE ?')
                    params.append(f'%"{tag}"%')
            
            # Add WHERE clause if needed
            if where_clauses:
                query += f' WHERE {" AND ".join(where_clauses)}'
            
            # Execute query
            cursor.execute(query, params)
            
            # Fetch results
            entries = []
            for row in cursor.fetchall():
                entry = {
                    'id': row[0],
                    'title': row[1],
                    'content': row[2],
                    'category': row[3],
                    'tags': json.loads(row[4]),
                    'source': row[5],
                    'timestamp': row[6],
                    'author_id': row[7],
                    'content_hash': row[8],
                    'metadata': json.loads(row[9])
                }
                entries.append(entry)
            
            conn.close()
            
            # Create export package
            export_data = {
                "export_date": datetime.now().isoformat(),
                "filters": {
                    "category": category,
                    "tags": tags
                },
                "entries_count": len(entries),
                "entries": entries
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting knowledge entries: {str(e)}")
            return {"error": str(e)}
    
    def import_entries(self, export_data: Dict[str, Any]) -> Tuple[int, int]:
        """
        Import knowledge entries from an export.
        
        Args:
            export_data: Export data from export_entries
            
        Returns:
            Tuple of (entries_imported, entries_skipped)
        """
        try:
            entries = export_data.get("entries", [])
            
            imported_count = 0
            skipped_count = 0
            
            for entry_data in entries:
                # Create entry object
                entry = KnowledgeEntry(
                    title=entry_data['title'],
                    content=entry_data['content'],
                    category=entry_data['category'],
                    tags=entry_data['tags'],
                    source=entry_data['source'],
                    author_id=entry_data['author_id'],
                    timestamp=datetime.fromisoformat(entry_data['timestamp']),
                    metadata=entry_data['metadata']
                )
                
                # Add to knowledge base
                result = self.add_entry(entry)
                
                if result > 0:
                    imported_count += 1
                else:
                    skipped_count += 1
            
            logger.info(f"Imported {imported_count} entries, skipped {skipped_count} entries")
            
            return (imported_count, skipped_count)
            
        except Exception as e:
            logger.error(f"Error importing knowledge entries: {str(e)}")
            return (0, 0)
    
    def rebuild_search_index(self) -> bool:
        """
        Rebuild the search index from scratch.
        
        Returns:
            True if rebuild was successful
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clear search index
            cursor.execute('DELETE FROM search_index')
            
            # Get all entries
            cursor.execute('''
                SELECT id, title, content, tags
                FROM knowledge_entries
            ''')
            
            # Fetch results
            entries = cursor.fetchall()
            
            # Rebuild index
            for entry_id, title, content, tags_json in entries:
                # Tokenize fields
                title_terms = self._tokenize_query(title)
                content_terms = self._tokenize_query(content)
                tags = json.loads(tags_json)
                
                # Add to index
                for term in set(title_terms + content_terms + tags):
                    cursor.execute('''
                        INSERT INTO search_index (entry_id, term)
                        VALUES (?, ?)
                    ''', (entry_id, term.lower()))
            
            # Commit and close
            conn.commit()
            conn.close()
            
            logger.info(f"Rebuilt search index with {len(entries)} entries")
            
            return True
            
        except Exception as e:
            logger.error(f"Error rebuilding search index: {str(e)}")
            return False
    
    def rebuild_vector_cache(self) -> bool:
        """
        Rebuild the vector cache for semantic search.
        
        Returns:
            True if rebuild was successful
        """
        try:
            # Ensure embedding model is loaded
            if not self._ensure_embedding_model():
                logger.warning("Embedding model not available, cannot rebuild vector cache")
                return False
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all entries
            cursor.execute('''
                SELECT id, title, content
                FROM knowledge_entries
            ''')
            
            # Fetch results
            entries = cursor.fetchall()
            conn.close()
            
            # Clear vector cache
            with self.vector_cache_lock:
                self.vector_cache.clear()
                
                # Rebuild cache
                for entry_id, title, content in entries:
                    # Get embedding
                    text = f"{title}\n{content}"
                    vector = self._get_embedding(text)
                    
                    # Add to cache
                    self.vector_cache[entry_id] = vector
            
            logger.info(f"Rebuilt vector cache with {len(entries)} entries")
            
            return True
            
        except Exception as e:
            logger.error(f"Error rebuilding vector cache: {str(e)}")
            return False
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database with required tables."""
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    source TEXT,
                    timestamp TEXT NOT NULL,
                    author_id INTEGER,
                    content_hash TEXT NOT NULL,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id INTEGER NOT NULL,
                    term TEXT NOT NULL,
                    FOREIGN KEY (entry_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE
                )
            ''')
            
            # Create indices
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entries_category ON knowledge_entries(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entries_content_hash ON knowledge_entries(content_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_term ON search_index(term)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_search_entry_id ON search_index(entry_id)')
            
            # Commit and close
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base database: {str(e)}")
            raise
    
    def _check_duplicate(self, entry: KnowledgeEntry) -> int:
        """
        Check if an entry is a duplicate of an existing entry.
        
        Args:
            entry: Entry to check
            
        Returns:
            ID of duplicate entry or 0 if not a duplicate
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for exact content hash match
            cursor.execute('''
                SELECT id FROM knowledge_entries
                WHERE content_hash = ?
            ''', (entry.content_hash,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return row[0]
            
            return 0
            
        except Exception as e:
            logger.error(f"Error checking for duplicate entry: {str(e)}")
            return 0
    
    def _update_search_index(self, entry: KnowledgeEntry) -> None:
        """
        Update the search index for an entry.
        
        Args:
            entry: Entry to update in the index
        """
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Remove existing index entries
            cursor.execute('DELETE FROM search_index WHERE entry_id = ?', (entry.entry_id,))
            
            # Tokenize fields
            title_terms = self._tokenize_query(entry.title)
            content_terms = self._tokenize_query(entry.content)
            
            # Add to index
            for term in set(title_terms + content_terms + entry.tags):
                cursor.execute('''
                    INSERT INTO search_index (entry_id, term)
                    VALUES (?, ?)
                ''', (entry.entry_id, term.lower()))
            
            # Commit and close
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating search index for entry {entry.entry_id}: {str(e)}")
    
    def _update_vector_cache(self, entry: KnowledgeEntry) -> None:
        """
        Update the vector cache for an entry.
        
        Args:
            entry: Entry to update in the cache
        """
        try:
            # Ensure embedding model is loaded
            if not self._ensure_embedding_model():
                return
            
            # Get embedding
            text = f"{entry.title}\n{entry.content}"
            vector = self._get_embedding(text)
            
            # Add to cache
            with self.vector_cache_lock:
                self.vector_cache[entry.entry_id] = vector
            
        except Exception as e:
            logger.error(f"Error updating vector cache for entry {entry.entry_id}: {str(e)}")
    
    def _tokenize_query(self, text: str) -> List[str]:
        """
        Tokenize text for search indexing.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace non-alphanumeric with spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split on whitespace
        tokens = text.split()
        
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'to', 'of', 'for', 'with', 'in', 'on', 'at',
            'by', 'this', 'that', 'these', 'those', 'it', 'its', 'from', 'as',
            'has', 'have', 'had', 'not', 'no', 'nor'
        }
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
        
        return tokens
    
    def _ensure_embedding_model(self) -> bool:
        """
        Ensure embedding model is loaded.
        
        Returns:
            True if embedding model is available
        """
        if self.embedding_model is not None:
            return True
        
        try:
            # Try to import sentence-transformers
            from sentence_transformers import SentenceTransformer
            
            # Load model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            return True
            
        except ImportError:
            logger.warning("sentence-transformers not available, semantic search will be disabled")
            return False
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            return False
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not self._ensure_embedding_model():
            return []
        
        # Get embedding
        embedding = self.embedding_model.encode(text)
        
        return embedding.tolist()
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity
        """
        try:
            import numpy as np
            
            # Convert to numpy arrays
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            similarity = dot_product / (norm_v1 * norm_v2)
            
            return float(similarity)
            
        except ImportError:
            logger.warning("numpy not available, using simplified similarity")
            
            # Simplified dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            
            # Simplified norms
            norm_v1 = sum(a * a for a in vec1) ** 0.5
            norm_v2 = sum(b * b for b in vec2) ** 0.5
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
                
            similarity = dot_product / (norm_v1 * norm_v2)
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def similarity_search_with_filter(self, 
                               query: str,
                               category: Optional[str] = None,
                               min_similarity: float = 0.6,
                               limit: int = 5) -> List[Tuple[KnowledgeEntry, float]]:
        """
        Perform semantic search with category filtering and similarity threshold.
        
        Args:
            query: Search query
            category: Category to filter by
            min_similarity: Minimum similarity threshold (0-1)
            limit: Maximum number of results to return
            
        Returns:
            List of tuples (entry, similarity_score)
        """
        # Get raw semantic search results
        results = self.semantic_search(query, limit=limit * 2)  # Get more results for filtering
        
        # Filter by category if specified
        if category:
            results = [(entry, score) for entry, score in results if entry.category == category]
        
        # Filter by minimum similarity
        results = [(entry, score) for entry, score in results if score >= min_similarity]
        
        # Return top results up to limit
        return results[:limit]

    def get_related_entries(self, entry_id: int, limit: int = 5) -> List[Tuple[KnowledgeEntry, float]]:
        """
        Get entries related to a specific entry.
        
        Args:
            entry_id: Entry ID to find related entries for
            limit: Maximum number of results to return
            
        Returns:
            List of tuples (entry, similarity_score)
        """
        # Get the entry
        entry = self.get_entry(entry_id)
        if not entry:
            return []
        
        # Create query from entry title and content
        query = f"{entry.title} {entry.content[:200]}"
        
        # Get similar entries
        results = self.semantic_search(query, limit=limit + 1)
        
        # Remove the original entry from results
        results = [(e, score) for e, score in results if e.entry_id != entry_id]
        
        # Return top results up to limit
        return results[:limit]

    def categorize_entry(self, content: str) -> str:
        """
        Automatically categorize content based on existing categories.
        
        Args:
            content: Text content to categorize
            
        Returns:
            Suggested category name
        """
        # Get existing categories
        categories = self.get_categories()
        
        if not categories:
            return "general"
        
        # If we have embedding model, use semantic similarity
        if self._ensure_embedding_model():
            # Embed the content
            content_embedding = self._get_embedding(content)
            
            # Compute similarity with each category
            category_scores = {}
            for category in categories:
                # Get sample entries for this category
                sample_entries = self.search(query="", category=category, limit=5)
                if not sample_entries:
                    continue
                
                # Get average embedding for category samples
                category_embeddings = []
                for entry in sample_entries:
                    entry_text = f"{entry.title} {entry.content}"
                    entry_embedding = self._get_embedding(entry_text)
                    category_embeddings.append(entry_embedding)
                
                if not category_embeddings:
                    continue
                    
                # Calculate average embedding
                avg_embedding = [sum(values) / len(values) for values in zip(*category_embeddings)]
                
                # Calculate similarity
                similarity = self._calculate_similarity(content_embedding, avg_embedding)
                category_scores[category] = similarity
            
            # Return category with highest similarity
            if category_scores:
                return max(category_scores.items(), key=lambda x: x[1])[0]
        
        # Fallback: return most common category
        return categories[0]

    def extract_tags_from_content(self, content: str) -> List[str]:
        """
        Extract relevant tags from content using NLP techniques.
        
        Args:
            content: Text content to extract tags from
            
        Returns:
            List of extracted tags
        """
        # Simple keyword extraction for now
        # In a more advanced implementation, this would use NLP techniques
        common_tags = self.get_tags()
        extracted_tags = []
        
        # Check for common tags in content
        for tag in common_tags:
            if tag.lower() in content.lower():
                extracted_tags.append(tag)
        
        # Limit to top 5 tags
        return extracted_tags[:5]

    def generate_knowledge_graph(self, root_entry_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a knowledge graph of connected entries.
        
        Args:
            root_entry_id: Optional root entry ID to start from
            
        Returns:
            Dictionary with nodes and edges for knowledge graph
        """
        nodes = []
        edges = []
        processed_ids = set()
        
        # Helper function to add entry and its related entries to graph
        def process_entry(entry_id, depth=0):
            if entry_id in processed_ids or depth > 2:  # Limit depth
                return
                
            processed_ids.add(entry_id)
            
            # Get entry
            entry = self.get_entry(entry_id)
            if not entry:
                return
                
            # Add node
            nodes.append({
                "id": str(entry.entry_id),
                "label": entry.title,
                "category": entry.category,
                "tags": entry.tags
            })
            
            # Get related entries
            related = self.get_related_entries(entry_id, limit=5)
            
            # Add edges and process related entries
            for related_entry, similarity in related:
                if similarity < 0.5:  # Minimum similarity threshold
                    continue
                    
                # Add edge
                edges.append({
                    "source": str(entry.entry_id),
                    "target": str(related_entry.entry_id),
                    "value": similarity
                })
                
                # Process related entry
                process_entry(related_entry.entry_id, depth + 1)
        
        # Start from root or process all entries
        if root_entry_id:
            process_entry(root_entry_id)
        else:
            # Get all entries (limit to 100 for performance)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM knowledge_entries LIMIT 100")
            entry_ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            for entry_id in entry_ids:
                process_entry(entry_id)
        
        return {
            "nodes": nodes,
            "edges": edges
        }

    def export_knowledge_base(self, format: str = 'json') -> str:
        """
        Export entire knowledge base to a file format.
        
        Args:
            format: Export format ('json', 'csv', 'markdown')
            
        Returns:
            String with exported content
        """
        # Get all entries
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, content, category, tags, source, timestamp,
                   author_id, content_hash, metadata
            FROM knowledge_entries
        """)
        
        entries = []
        for row in cursor.fetchall():
            entry = {
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'category': row[3],
                'tags': json.loads(row[4]),
                'source': row[5],
                'timestamp': row[6],
                'author_id': row[7],
                'content_hash': row[8],
                'metadata': json.loads(row[9])
            }
            entries.append(entry)
        
        conn.close()
        
        if format == 'json':
            return json.dumps({
                "export_date": datetime.utcnow().isoformat(),
                "entries_count": len(entries),
                "entries": entries
            }, indent=2)
            
        elif format == 'csv':
            import csv
            from io import StringIO
            
            output = StringIO()
            fieldnames = ['id', 'title', 'category', 'tags', 'source', 'timestamp', 'author_id']
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in entries:
                writer.writerow({
                    'id': entry['id'],
                    'title': entry['title'],
                    'category': entry['category'],
                    'tags': ','.join(entry['tags']),
                    'source': entry['source'],
                    'timestamp': entry['timestamp'],
                    'author_id': entry['author_id']
                })
            
            return output.getvalue()
            
        elif format == 'markdown':
            md_lines = ["# Knowledge Base Export", f"Generated on: {datetime.utcnow().isoformat()}", ""]
            
            # Group by category
            by_category = {}
            for entry in entries:
                category = entry['category']
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(entry)
            
            # Generate markdown
            for category, category_entries in by_category.items():
                md_lines.append(f"## {category}")
                md_lines.append("")
                
                for entry in category_entries:
                    md_lines.append(f"### {entry['title']}")
                    if entry['tags']:
                        md_lines.append(f"*Tags: {', '.join(entry['tags'])}*")
                    md_lines.append("")
                    md_lines.append(entry['content'])
                    md_lines.append("")
                    md_lines.append(f"*Source: {entry['source'] or 'Unknown'} | Created: {entry['timestamp']}*")
                    md_lines.append("---")
                    md_lines.append("")
            
            return "\n".join(md_lines)
            
        else:
            return f"Unsupported export format: {format}"

# Initialize the global knowledge base
knowledge_base = KnowledgeBase()