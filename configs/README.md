# Configs

This directory contains configuration files for the SCAFAD Delta system.

## Overview

The `configs` folder houses configuration modules that define system parameters, settings, and operational configurations for the SCAFAD Delta framework.

## Contents

- **layer1_config.py** - Core configuration module for Layer 1 operations, including system parameters, thresholds, and operational settings

## Usage

Configuration files in this directory are imported by core modules to establish system behavior and operational parameters. They provide centralized management of system settings and can be modified to adjust system behavior without changing core logic.

## Configuration Management

- All configuration values are centralized in this directory
- Configuration files support environment-specific overrides
- Changes to configuration files require system restart to take effect
- Configuration validation is performed at startup

## Related Documentation

- See `docs/architecture.md` for system architecture details
- See `docs/api-reference.md` for configuration API usage
