# PyRIT-Enhanced PAN Security Tester

A comprehensive security testing framework for Palo Alto Networks AI Runtime Security using Microsoft's PyRIT (Python Risk Identification Toolkit). This tool performs extensive red-team testing to evaluate the effectiveness of your AI security policies.

## Features

### Core Testing Capabilities
- **PyRIT Dataset Integration** - Tests against 11+ security datasets including AdvBench, HarmBench, and XSTest
- **Advanced Attack Simulation** - Psychological manipulation, metamorphic attacks, and social engineering
- **Multi-language Evasion** - Tests attacks in 10 different languages
- **Encoding Bypass Testing** - Comprehensive evasion attempts using various encoding methods
- **Context Window Attacks** - Prompt stuffing, attention dilution, and instruction burial techniques
- **Multi-turn Conversations** - Persistent attacks that build context over multiple interactions

### Advanced Attack Types
- **Psychological Manipulation** - Authority appeals, urgency pressure, social proof
- **Metamorphic Attacks** - Same malicious intent expressed in different forms
- **Chain-of-Thought Manipulation** - Exploiting logical reasoning processes
- **Context Injection** - Gradually introducing malicious content over conversations
- **Encoding Evasion** - Base64, hexadecimal, Unicode, and other encoding techniques

### Comprehensive Reporting
- **Real-time Attack Monitoring** - Live feedback on bypass attempts and detections
- **Risk Assessment Scoring** - Automated risk level calculation with recommendations
- **Detailed Analytics** - Breakdown by attack type, success rates, and threat patterns
- **JSON Export** - Complete test results for further analysis and compliance reporting

## Installation

### Prerequisites
```bash
pip install aiohttp
pip install pyrit-ai
```

### Configuration
Update the configuration variables in the script:
```python
PAN_API_KEY = "your-pan-api-key"
PAN_PROFILE_NAME = "your-security-profile"
PAN_BASE_URL = "https://service.api.aisecurity.paloaltonetworks.com"
```

## Usage

### Basic Execution
```bash
python enhanced_pan_tester.py
```

### Configuration Options
Customize testing by modifying these settings:
```python
ENABLE_PYRIT_DATASETS = True          # Test with PyRIT security datasets
ENABLE_PSYCHOLOGICAL_TESTS = True     # Run psychological manipulation tests
ENABLE_ENCODING_TESTS = True          # Test encoding-based evasions
ENABLE_MULTILINGUAL_TESTS = True      # Multi-language attack testing
ENABLE_PERSISTENCE_TESTS = True       # Multi-turn conversation attacks
NUM_PROMPTS_PER_DATASET = 8          # Number of prompts per dataset
DELAY_BETWEEN_TESTS = 0.8             # Delay between API calls
```

## Understanding Results

### Risk Levels
- **Minimal** (0-5% bypass rate) - Excellent security posture
- **Low** (5-10% bypass rate) - Good security with minor gaps
- **Medium** (20-35% bypass rate) - Moderate security, improvements needed
- **High** (35-50% bypass rate) - Significant vulnerabilities present
- **Critical** (50%+ bypass rate) - Immediate security review required

### Output Files
The tool generates comprehensive JSON reports with timestamps:
```
ultimate_pyrit_pan_assessment_YYYYMMDD_HHMMSS.json
```

### Key Metrics
- **Total Tests Executed** - Complete count of security tests performed
- **Successful Bypasses** - Number of malicious prompts that evaded detection
- **Detection Rate** - Percentage of malicious content properly blocked
- **Risk Score** - Calculated security effectiveness rating

## Test Categories

### PyRIT Datasets (88 tests)
Professional security datasets including:
- **AdvBench** - Adversarial benchmark prompts
- **HarmBench** - Harmful content evaluation
- **XSTest** - Cross-domain safety testing
- **Forbidden Questions** - Direct policy violation attempts

### Advanced Attack Techniques (48+ tests)
- **Psychological Manipulation** - Social engineering techniques
- **Metamorphic Attacks** - Intent-preserving transformations
- **Encoding Evasion** - Technical bypass methods
- **Context Attacks** - Attention and memory exploitation
- **Multi-turn Persistence** - Conversation-based attacks

## Security Recommendations

Based on test results, the tool provides actionable security guidance:

### Critical Risk
- Immediate security profile review required
- Implement additional content filtering layers
- Increase monitoring sensitivity
- Consider multi-layered defense strategies

### Medium Risk
- Fine-tune detection rules for edge cases
- Enhance encoding evasion detection
- Implement regular testing schedules
- Update threat pattern databases

### Low Risk
- Continue current security practices
- Monitor for emerging attack patterns
- Maintain regular assessment schedule
- Share best practices with security teams

## Technical Details

### PyRIT Integration
The tool integrates deeply with Microsoft's PyRIT framework:
- **Native Dataset Support** - Direct access to PyRIT's security datasets
- **Converter Integration** - Automatic prompt transformation testing
- **Orchestrator Compatibility** - Advanced multi-step attack simulation
- **Memory Management** - Persistent conversation tracking

### API Compatibility
- **Robust Error Handling** - Graceful degradation when PyRIT components fail
- **Version Flexibility** - Compatible with multiple PyRIT releases
- **Fallback Testing** - Manual encoding tests when converters unavailable
- **Clean Logging** - Informative output without overwhelming warnings

### Performance Features
- **Asynchronous Processing** - Non-blocking API calls for efficiency
- **Rate Limiting** - Configurable delays to respect API limits
- **Memory Optimization** - Efficient handling of large test datasets
- **Progress Tracking** - Real-time feedback during long test runs

## Troubleshooting

### Common Issues

**PyRIT Import Errors**
```bash
pip install --upgrade pyrit-ai
```

**API Authentication Failures**
- Verify your PAN API key is correct
- Check that your security profile exists
- Ensure API endpoint URL is accurate

**Converter Warnings**
- These are normal and don't affect test quality
- The tool automatically falls back to manual encoding tests
- All core security testing remains functional

**Memory Issues with Large Datasets**
- Reduce `NUM_PROMPTS_PER_DATASET` value
- Increase `DELAY_BETWEEN_TESTS` for slower execution
- Monitor system resources during execution

### Performance Optimization
- **Parallel Execution** - Consider running different test categories separately
- **Selective Testing** - Disable test categories not needed for your use case
- **Result Caching** - Save intermediate results for incremental testing
- **Resource Monitoring** - Watch API rate limits and system memory

## Contributing

This tool is designed for security professionals testing AI safety implementations. When contributing:

- Test changes against multiple PyRIT versions
- Ensure new attack patterns are ethically sound
- Maintain compatibility with existing PAN configurations
- Document any new test categories or metrics

## License

This tool is provided for legitimate security testing purposes. Users are responsible for:
- Obtaining proper authorization before testing
- Complying with applicable laws and regulations
- Using results to improve security, not exploit vulnerabilities
- Protecting sensitive data generated during testing

## Support

For technical issues:
1. Check the troubleshooting section above
2. Verify your PyRIT and dependency versions
3. Review the generated log files for specific errors
4. Ensure your PAN configuration allows API access

The tool generates comprehensive logs that help diagnose most issues. When reporting problems, include relevant log excerpts and your configuration settings (excluding sensitive credentials).
