"""
Batch Processing Framework for PhysioRAG

Handles efficient batch processing of questions and answers with:
- Checkpointing and resume capability
- Batching of requests
- Progress tracking
- Error handling and logging
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime
import traceback
from config import BATCH_SIZE, CHECKPOINT_INTERVAL, RESULTS_DIR

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Generic batch processor with checkpoint/resume capability."""

    def __init__(
        self,
        job_name: str,
        output_dir: str = RESULTS_DIR,
        batch_size: int = BATCH_SIZE,
        checkpoint_interval: int = CHECKPOINT_INTERVAL,
    ):
        """
        Initialize batch processor.

        Args:
            job_name: Identifier for this batch job
            output_dir: Directory to save results and checkpoints
            batch_size: Number of items to process per batch
            checkpoint_interval: Save checkpoint every N items
        """
        self.job_name = job_name
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint file
        self.checkpoint_file = self.output_dir / f"{job_name}_checkpoint.json"
        self.results_file = self.output_dir / f"{job_name}_results.json"
        self.log_file = self.output_dir / f"{job_name}_log.txt"

        # Setup logging
        self._setup_logging()

        logger.info(f"Initialized BatchProcessor: {job_name}")

    def _setup_logging(self) -> None:
        """Setup file logging for batch processing."""
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load checkpoint if it exists.

        Returns:
            Checkpoint dict with processed items and current progress
        """
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                logger.info(
                    f"Loaded checkpoint: {len(checkpoint.get('results', []))} items processed"
                )
                return checkpoint
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                return {"results": [], "processed_count": 0}
        return {"results": [], "processed_count": 0}

    def save_checkpoint(self, results: List[Dict], processed_count: int) -> None:
        """
        Save checkpoint with current progress.

        Args:
            results: List of processed results
            processed_count: Total number of items processed
        """
        checkpoint = {
            "job_name": self.job_name,
            "timestamp": datetime.now().isoformat(),
            "processed_count": processed_count,
            "results": results,
        }
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
            logger.info(f"Checkpoint saved: {processed_count} items processed")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def save_results(self, results: List[Dict], metadata: Optional[Dict] = None) -> Path:
        """
        Save final results to JSON file.

        Args:
            results: List of processed results
            metadata: Optional metadata to include in results file

        Returns:
            Path to results file
        """
        output = {
            "job_name": self.job_name,
            "timestamp": datetime.now().isoformat(),
            "total_count": len(results),
            "metadata": metadata or {},
            "results": results,
        }
        try:
            with open(self.results_file, 'w') as f:
                json.dump(output, f, indent=2, default=str)
            logger.info(f"Results saved to: {self.results_file}")
            return self.results_file
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    def process_batch(
        self,
        items: List[Any],
        process_func: Callable[[Any], Dict],
        resume: bool = False,
    ) -> List[Dict]:
        """
        Process items in batches with checkpoint capability.

        Args:
            items: List of items to process
            process_func: Function to apply to each item
            resume: Whether to resume from checkpoint

        Returns:
            List of processed results
        """
        # Load checkpoint if resuming
        checkpoint = self.load_checkpoint() if resume else {"results": [], "processed_count": 0}
        results = checkpoint.get("results", [])
        start_idx = checkpoint.get("processed_count", 0)

        logger.info(f"Starting batch processing: {len(items)} items total, resuming from {start_idx}")

        try:
            for idx in range(start_idx, len(items)):
                item = items[idx]

                try:
                    # Process the item
                    result = process_func(item)
                    result["_index"] = idx
                    result["_status"] = "success"
                    results.append(result)

                except Exception as e:
                    # Log error but continue processing
                    logger.error(f"Error processing item {idx}: {str(e)}")
                    logger.debug(traceback.format_exc())
                    results.append({
                        "_index": idx,
                        "_status": "error",
                        "_error": str(e),
                    })

                # Log progress
                if (idx + 1) % 10 == 0:
                    logger.info(f"Progress: {idx + 1}/{len(items)} items processed")

                # Save checkpoint periodically
                if (idx + 1) % self.checkpoint_interval == 0:
                    self.save_checkpoint(results, idx + 1)

            # Final checkpoint
            self.save_checkpoint(results, len(items))
            logger.info(f"Batch processing completed: {len(results)} results")
            return results

        except KeyboardInterrupt:
            logger.warning("Batch processing interrupted by user")
            self.save_checkpoint(results, len(results))
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        checkpoint = self.load_checkpoint()
        return {
            "job_name": self.job_name,
            "processed_count": checkpoint.get("processed_count", 0),
            "checkpoint_file": str(self.checkpoint_file),
            "results_file": str(self.results_file),
            "checkpoint_exists": self.checkpoint_file.exists(),
            "results_exists": self.results_file.exists(),
        }


class QuestionAnswerBatchProcessor(BatchProcessor):
    """Specialized batch processor for question/answer generation."""

    def __init__(
        self,
        job_name: str,
        retriever,
        generator,
        output_dir: str = RESULTS_DIR,
        batch_size: int = BATCH_SIZE,
        checkpoint_interval: int = CHECKPOINT_INTERVAL,
    ):
        """
        Initialize QA batch processor.

        Args:
            job_name: Identifier for this batch job
            retriever: MedicalHybridRetriever instance
            generator: ReasoningGenerator instance
            output_dir: Directory to save results
            batch_size: Number of items per batch
            checkpoint_interval: Checkpoint save interval
        """
        super().__init__(
            job_name=job_name,
            output_dir=output_dir,
            batch_size=batch_size,
            checkpoint_interval=checkpoint_interval,
        )
        self.retriever = retriever
        self.generator = generator

    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a single question: retrieve context and generate answer.

        Args:
            question: The question to process

        Returns:
            Dict with question, retrieved contexts, and answer
        """
        try:
            # Retrieve relevant documents
            context_docs = self.retriever.get_relevant_documents(question)

            # Generate answer
            answer = self.generator.generate_answer(question, context_docs)

            # Extract metadata from retrieved docs
            sources = [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "section": doc.metadata.get("section", "General"),
                }
                for doc in context_docs
            ]

            return {
                "question": question,
                "answer": answer,
                "retrieved_docs_count": len(context_docs),
                "sources": sources,
                "_status": "success",
            }

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "question": question,
                "_status": "error",
                "_error": str(e),
            }

    def process_questions(
        self,
        questions: List[str],
        resume: bool = False,
    ) -> List[Dict]:
        """
        Process a batch of questions.

        Args:
            questions: List of questions to process
            resume: Whether to resume from checkpoint

        Returns:
            List of question/answer results
        """
        return self.process_batch(
            items=questions,
            process_func=self.process_question,
            resume=resume,
        )


