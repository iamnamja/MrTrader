import json
import logging
from typing import Any, Optional, Dict, List
import redis
from datetime import datetime
from app.config import settings

logger = logging.getLogger(__name__)


class RedisQueue:
    """Redis-based message queue for agent communication"""

    def __init__(self, url: str = None):
        self.url = url or settings.redis_url
        try:
            self.redis_client = redis.from_url(
                self.url, decode_responses=True,
                socket_connect_timeout=3,
                # No socket_timeout here — blpop needs to hold the socket open for its full timeout duration
            )
            self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def push(self, queue_name: str, message: Dict[str, Any]) -> bool:
        """
        Push a message to a queue

        Args:
            queue_name: Name of the queue
            message: Message dictionary
        """
        try:
            # Add timestamp if not present
            if "timestamp" not in message:
                message["timestamp"] = datetime.utcnow().isoformat()

            json_message = json.dumps(message)
            self.redis_client.rpush(queue_name, json_message)
            logger.debug(f"Pushed message to {queue_name}: {message}")
            return True
        except Exception as e:
            logger.error(f"Error pushing to queue {queue_name}: {e}")
            return False

    def pop(self, queue_name: str, timeout: int = 1) -> Optional[Dict[str, Any]]:
        """
        Pop a message from a queue (blocking)

        Args:
            queue_name: Name of the queue
            timeout: Blocking timeout in seconds
        """
        try:
            result = self.redis_client.blpop(queue_name, timeout=timeout)
            if result:
                _, json_message = result
                message = json.loads(json_message)
                logger.debug(f"Popped message from {queue_name}: {message}")
                return message
            return None
        except Exception as e:
            logger.error(f"Error popping from queue {queue_name}: {e}")
            return None

    def get_queue_length(self, queue_name: str) -> int:
        """Get number of messages in a queue"""
        try:
            return self.redis_client.llen(queue_name)
        except Exception as e:
            logger.error(f"Error getting queue length: {e}")
            return 0

    def peek_queue(self, queue_name: str, count: int = 10) -> List[Dict[str, Any]]:
        """Peek at messages in a queue without removing them"""
        try:
            messages = []
            items = self.redis_client.lrange(queue_name, 0, count - 1)
            for item in items:
                messages.append(json.loads(item))
            return messages
        except Exception as e:
            logger.error(f"Error peeking queue: {e}")
            return []

    def clear_queue(self, queue_name: str) -> bool:
        """Clear all messages from a queue"""
        try:
            self.redis_client.delete(queue_name)
            logger.info(f"Cleared queue: {queue_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing queue: {e}")
            return False

    def set_state(self, key: str, value: Any, expire: int = None) -> bool:
        """
        Store state/data in Redis

        Args:
            key: State key
            value: State value (will be JSON serialized)
            expire: Expiration time in seconds
        """
        try:
            json_value = json.dumps(value) if not isinstance(value, str) else value
            if expire:
                self.redis_client.setex(key, expire, json_value)
            else:
                self.redis_client.set(key, json_value)
            logger.debug(f"Set state: {key}")
            return True
        except Exception as e:
            logger.error(f"Error setting state: {e}")
            return False

    def get_state(self, key: str) -> Optional[Any]:
        """Retrieve state/data from Redis"""
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting state: {e}")
            return None

    def delete_state(self, key: str) -> bool:
        """Delete state/data from Redis"""
        try:
            self.redis_client.delete(key)
            logger.debug(f"Deleted state: {key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting state: {e}")
            return False

    def increment_counter(self, key: str, amount: int = 1) -> int:
        """Increment a counter"""
        try:
            return self.redis_client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Error incrementing counter: {e}")
            return 0

    def health_check(self) -> bool:
        """Check if Redis is accessible using a short-timeout probe connection."""
        try:
            import redis as _redis
            probe = _redis.from_url(
                self.url, decode_responses=True,
                socket_timeout=2, socket_connect_timeout=2,
            )
            probe.ping()
            probe.close()
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False


# Global instance
redis_queue = None


def get_redis_queue() -> RedisQueue:
    """Get or create Redis queue instance"""
    global redis_queue
    if redis_queue is None:
        redis_queue = RedisQueue()
    return redis_queue
