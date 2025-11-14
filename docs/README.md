# Documentation Directory

This directory contains all project documentation for the DLLM Delta-Compute project.

## Structure

```
docs/
├── master_plan.md                    # **PRIMARY**: Complete project specification (P0-P4, Phase A-C)
├── implementation_summary.md         # **CURRENT**: Implementation details and architecture
├── reports/                          # Validation and evaluation reports
│   └── GPU_VALIDATION_REPORT.md     # H100 GPU validation results
├── archive/                          # Historical documents (for reference only)
│   ├── development_log.md           # Development progress log (archived)
│   ├── feasibility_re_evaluation.md # Initial feasibility analysis (archived)
│   ├── implementation_issues.md     # Early implementation notes (archived)
│   ├── development_issues.md        # Issue tracking (archived)
│   └── session_summary_2025-11-14.md # Session notes (archived)
└── material/                         # Reference materials (gitignored)
```

## Key Documents

### Active Documents (Always Current)

1. **`master_plan.md`** ⭐ **PRIMARY SPECIFICATION**
   - Complete project scope: ALL P0-P4 and Phase A-C phases
   - Architecture constraints and design decisions
   - Evaluation tasks and metrics
   - Deliverables and completion criteria
   - **Status**: Living document, updated as project evolves

2. **`implementation_summary.md`** 📋 **IMPLEMENTATION REFERENCE**
   - Current implementation status (~80% complete)
   - Modified files and code locations
   - API changes and integration points
   - Testing instructions
   - **Status**: Updated after major implementation milestones

3. **`reports/GPU_VALIDATION_REPORT.md`** ✅ **VALIDATION RESULTS**
   - H100 80GB GPU validation results
   - Environment setup verification
   - Known issues and next steps
   - **Status**: Updated after validation runs

### Archived Documents (Historical Reference)

Located in `archive/` directory. These are kept for historical context but are **not actively maintained**:

- `development_log.md` - Early development progress tracking
- `feasibility_re_evaluation.md` - Initial feasibility analysis (Nov 14)
- `implementation_issues.md` - Early implementation challenges
- `development_issues.md` - Issue tracking (superseded by git commits)
- `session_summary_2025-11-14.md` - Session notes from initial setup

## Documentation Workflow

### When to Update Which Document

1. **Changing project scope or requirements** → Update `master_plan.md`
2. **Completing a phase (P0, P1, P2, etc.)** → Update both `master_plan.md` (progress) and `implementation_summary.md` (details)
3. **Running validation/evaluation** → Create new report in `reports/`
4. **Daily development notes** → Use git commit messages, not separate docs

### Creating New Reports

Evaluation and validation reports should go in `reports/` with naming convention:
- `{phase}_{task}_{date}_report.md`
- Examples:
  - `P0_baseline_gsm8k_2025-11-15_report.md`
  - `P2_early_stop_gsm8k_2025-11-20_report.md`
  - `PhaseA_caching_gsm8k_2025-11-25_report.md`

## Quick Reference

**"Which doc should I read first?"**
- For project overview and scope → `master_plan.md`
- For implementation details → `implementation_summary.md`
- For current status → Section 10 of `master_plan.md` + latest report in `reports/`

**"Where do I track progress?"**
- Implementation progress → `master_plan.md` Section 10
- Code changes → Git commit history
- Evaluation results → `reports/` directory

**"Are old docs still valid?"**
- `archive/` contents are historical only
- May contain outdated information
- Use for understanding project evolution, not current status
