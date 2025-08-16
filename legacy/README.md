# Legacy

This directory contains legacy code and deprecated components from previous versions of the SCAFAD Delta system.

## Overview

The `legacy` folder houses older implementations and deprecated functionality that are maintained for backward compatibility and reference purposes.

## Contents

- **fusion_app.py** - Legacy fusion application implementation

## Purpose

This directory serves several purposes:
- Maintaining backward compatibility for existing deployments
- Providing reference implementations for migration
- Preserving historical functionality for analysis
- Supporting gradual migration to new architectures

## Migration Guidelines

### When to Use Legacy Code
- Existing production systems requiring backward compatibility
- Reference implementations for understanding old patterns
- Migration planning and analysis

### Migration Path
1. Identify functionality needed from legacy code
2. Review equivalent features in current system
3. Plan migration strategy and timeline
4. Implement new functionality
5. Test thoroughly before removing legacy code

## Deprecation Status

Legacy components are marked as deprecated and will be removed in future versions:
- **fusion_app.py** - Deprecated, use core modules instead

## Maintenance

Legacy code is maintained with minimal updates:
- Critical bug fixes only
- No new feature development
- Limited testing coverage
- Documentation may be outdated

## Related Resources

- See `core/` directory for current implementations
- See `docs/architecture.md` for current system design
- See `examples/` directory for current usage patterns
