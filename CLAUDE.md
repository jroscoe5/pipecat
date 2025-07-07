# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Pipecat is an open-source Python framework for building real-time voice and multimodal conversational AI agents. It provides a frame-based pipeline architecture for orchestrating audio/video, AI services, different transports, and conversation pipelines.

## Common Development Commands

### Installation and Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r dev-requirements.txt

# Install pre-commit hooks
pre-commit install

# Install pipecat locally in editable mode
pip install -e .

# Install with optional dependencies (examples)
pip install -e ".[daily,deepgram,cartesia,openai,silero]"
```

### Build, Test, and Lint

```bash
# Run tests
pytest

# Run specific test
pytest tests/test_pipeline.py::TestClassName::test_method_name

# Format code
ruff format

# Check linting
ruff check

# Run both formatting and linting checks (pre-commit script)
./scripts/pre-commit.sh

# Build the package
python -m build .
```

## High-Level Architecture

### Frame-Based Data Flow

The core of Pipecat is its frame system. All data and control information flows through the pipeline as typed frames:

- **Frame**: Base class with ID, name, timestamps, and metadata
- **SystemFrame**: Immediate processing (bypasses queuing) - e.g., StartFrame, CancelFrame, interruptions
- **DataFrame**: Ordered data frames - audio, images, text, LLM messages
- **ControlFrame**: Ordered control frames - EndFrame, HeartbeatFrame, TTS controls

### Pipeline Architecture

Pipelines connect frame processors in sequence or parallel:

1. **Linear Pipeline**: Sequential processing with automatic source/sink management
2. **Parallel Pipeline**: Concurrent processing with synchronization on EndFrame
3. **Frame Direction**: DOWNSTREAM (input→output) and UPSTREAM (control signals)

### Processor Model

All components inherit from `FrameProcessor`:
- Asynchronous frame processing with input/output queues
- Lifecycle management (setup/cleanup)
- Interruption handling
- Metrics collection support
- System frames bypass queuing for immediate processing

### AI Service Integration

Services (LLM, TTS, STT, Vision) extend `AIService`:
- Standardized lifecycle methods: start(), stop(), cancel()
- Dynamic settings updates
- Model switching capabilities
- Generator support for async frame streams

## Code Style Guidelines

- Follow Google-style docstrings
- Use Ruff for formatting (line length: 100)
- All public methods must have docstrings with Args: and Returns: sections
- Async-first design for all processing components
- Type hints required for all function parameters and returns

## Project Structure

```
src/pipecat/
├── frames/          # Frame definitions and protobuf serialization
├── processors/      # Core processing components
├── pipeline/        # Pipeline orchestration
├── services/        # AI service integrations
├── transports/      # Communication layers (WebRTC, WebSocket)
├── audio/           # Audio utilities
├── metrics/         # Metrics collection
└── utils/           # Common utilities

examples/
├── foundational/    # Step-by-step learning examples
└── [app-examples]/  # Complete application examples
```

## Testing Approach

- Unit tests use pytest with async support
- Test files mirror source structure in tests/
- Mock AI services for unit testing
- Integration tests for service interactions
- Coverage reports via codecov

## Key Design Patterns

1. **Frame-based Communication**: All inter-component communication via typed frames
2. **Pipeline Composition**: Processors compose into pipelines; pipelines are processors
3. **Asynchronous Processing**: Task-based architecture with proper lifecycle management
4. **Interruption Handling**: Built-in support with multiple strategies
5. **Metrics and Observability**: Integrated at processor level with TTFB tracking