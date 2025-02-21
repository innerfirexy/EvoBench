# LLM-Generated Text Detection Benchmarking for Evolving Large Language Models
<!-- 
| Detection Method | GPT-3.5 | GPT-4 | Claude | LLaMA | Performance Drop (%) |
|------------------|---------|-------|--------|-------|---------------------|
| Method A         | 92.3%   | 78.6% | 82.1%  | 81.5% | -13.7%              |
| Method B         | 89.8%   | 75.2% | 80.3%  | 77.6% | -14.6%              |
| Method C         | 94.1%   | 81.2% | 84.7%  | 80.3% | -12.9%              |
| Method D         | 90.5%   | 76.8% | 79.5%  | 75.2% | -15.3%              |
| Watermarking     | 96.2%   | 68.7% | 72.1%  | 67.5% | -27.5%              | -->

## 💡 Introduction
We introduce **EvoBench**, a dynamic benchmark considering a new dimension of generalization across continuously evolving LLMs. EvoBench categorizes the evolving LLMs into (1) **updates over time** and (2) **developments like finetuning and pruning**, covering 7 LLM families (Claude, Gemini, GPT4, GPT4o, LlaMA-2-7B, LlaMA3, and Qwen) and their evolving versions.

## 🔥 News

- **[Feb 2025]** EvoBench v1.0 released with support for 30 evolving LLMs across 7 LLM families.
- **[Jan 2025]** First LLM-generated Text Detection Benchmark for evolving LLMs available for research use.

## 📊 Benchmark Design & Performance

### LLM Evolution Coverage

EvoBench comprehensively assesses generalization across evolving LLMs by including:

- **7 widely used LLM families**: GPT-4o, GPT-4, Claude, Gemini, Qwen, LlaMA3, and LlaMA2
- **30 evolving versions** spanning these families
- **Updates over time**: Tracking performance across model updates for GPT-4o, GPT-4, Claude, Gemini, Qwen, and LlaMA3
- **Development variations**: Focusing on the LlaMA2 family with fine-tuning and pruning techniques

### Domain Generalization

EvoBench includes five datasets spanning diverse domains:

- **XSum**: News articles summarization
- **WritingPrompts**: Creative story writing 
- **PubMed**: Biomedical research question answering
- **PeerRead**: Peer-reviewed academic writing
- **SocialMedia**: Social media content

### Generation Paradigm Diversity

The benchmark incorporates three distinct generation paradigms:

1. **Continuation-based generation**: XSum, WritingPrompts, and PeerRead
2. **Question-answering generation**: PubMed
3. **Paraphrased generation**: SocialMedia

## 📁 Repository Structure

```
EvoBench_happy_moer/
├── Claude/                    # Claude model family
│   ├── Claude-Haiku/          # Claude-Haiku model versions
│   │   ├── args/              # Arguments and configuration
│   │   ├── harmful_*.json     # Social media content domain data
│   │   ├── peerread_*.json    # Peer-reviewed academic writing data
│   │   ├── pubmed_*.json      # Biomedical research Q&A data
│   │   ├── writing_*.json     # Creative story writing data
│   │   └── xsum_*.json        # News articles summarization data
│   ├── Claude-Sonnet/         # Claude-Sonnet model versions
│   └── ...                    # Other Claude model variants
├── Gemini/                    # Gemini model family
├── GPT4/                      # GPT-4 model family (updates over time)
├── GPT4o/                     # GPT-4o model family (updates over time)
├── LlaMA-2-7B/                # LlaMA-2-7B model family (development variations)
│   └── ...                    # Fine-tuned and pruned versions
├── LlaMA3/                    # LlaMA3 model family (updates over time)
└── Qwen/                      # Qwen model family (updates over time)
```

Each model family contains multiple evolving versions across different generations, tested on five domains using three generation paradigms:
1. Continuation-based: XSum, Writing, PeerRead
2. Question-answering: PubMed
3. Paraphrased: SocialMedia (harmful)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, please open an issue on this repository or contact us at yuxiao1217@mail.ustc.edu.cn.



