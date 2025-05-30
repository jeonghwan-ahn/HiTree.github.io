---
layout: distill
title: HiTree:Efficient Hierarchical Speculative Decoding via Tree-Based Drafting
description: To conduct the LLM inference efficiently, speculative decoding is emerging as a promising solution by using a small draft model to propose tokens and a large target model to verify them. In this project, we propose to extend and improve the hierarchical speculative decoding method by integrating a tree-based architecture. This extension aims to support multi-sequence inference and explore hierarchical token prediction more efficiently. Our approach focuses on expanding the scalability of speculative decoding while preserving or improving accuracy and inference latency.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# anonymize when submitting 
authors:
  - name: Jiwoo Kim
    #url: "https://www.linkedin.com/in/amitoj-singh-miglani-76baa2291/"
    affiliations:
      name: POSTECH, Korea
  - name: Vidit Aggarwal
    #url: "https://www.linkedin.com/in/vidit-aggarwal-328525285/"
    affiliations:
      name: POSTECH, Korea



bibliography: 2025-04-28-analysing-the-spectral-biases-in-generative-models.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Viewing Images in Frequency Domain
  - name: Analysis of Bias in GANs
  # you can additionally add subentries like so
    subsections:
    - name: Setting Up the Generative CNN Structure
    - name: ReLU as a Fixed Binary Mask
    - name: Onto The Analysis of Filter Spectrum
  - name: Frequency bias in Diffusion Models
  - name: Mitigation of Frequency Bias Using Spectral Diffusion Model
  - name: Conclusion
---



# HiTree: Efficient Hierarchical Speculative Decoding via Tree-Based Drafting
Large Language Models (LLMs) have enabled remarkable progress in diverse domains by delivering high-quality and precise text generation. Despite these advances, the process of text generation remains computationally intensive and time-consuming, posing challenges for efficient and scalable deployment. 

To overcome these hurdles, innovative techniques like speculative decoding have emerged. This approach leverages a smaller draft model to quickly propose candidate tokens, which are then verified by a larger, more accurate target model.

In this blog post, we will explore the fundamental bottlenecks that hinder efficient LLM text generation and introduce the concept of speculative decoding in detail.Additionally, we specifically explore hierarchical approaches to speculative decoding and propose enhancing their performance by transforming into tree-based architectures.


## Why is LLM Generation Inefficient?

### LLM Inference Phases
The LLM inference process is broadly divided into two stages. The first is the prefill stage, where all input tokens are processed in parallel to generate the initial output token. This stage involves a large amount of matrix operations and attention computations, making it computationally intensive. Additionally, the context generated from the attention layers is stored in the key-value (KV) cache and used for subsequent token generation. For these reasons, the prefill stage is considered a computation-bound phase that mainly relies on processing power.

The second stage is the decode phase, which begins after the first token is generated. In this phase, tokens are generated autoregressively, with each new token predicted based only on the previously generated token and the KV cache. Because tokens are produced one at a time, parallel processing is not possible, and frequent access to the KV cache places a heavy demand on memory bandwidth and capacity. Therefore, the decode phase is characterized as a memory-bound stage, where memory access becomes the main bottleneck rather than computation.

### Bottlenecks in LLM Inference
From a system perspective, most of the latency in LLM inference is known to occur during the decode phase[^splitwise]. In other words, the primary bottleneck arises from the accelerator’s memory bandwidth limitations rather than arithmetic computations. This is due to the sequential nature of auto-regressive decoding, where each forward pass requires transferring the entire model parameters from High-Bandwidth Memory (HBM) to the accelerator’s cache. Because tokens are generated one at a time, the high computational capabilities of modern accelerators are underutilized, leading to inefficiencies.

This inefficiency is clearly reflected in cost structures as well. As of September 2023, the generation cost for GPT-4 is approximately twice that of simple prompt processing, while for Claude 2, it is about three times higher. The need for improvement in the decoding phase is clearly evident, and speculative decoding has been gaining attention as a promising technique to address this.


<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/SJZOS3Sfgg.jpg" alt="per token price" width="600"/>
  <p style="font-size: 0.9em; color: gray;">
    Figure 1: Per token price comparison of GPT-4 and Claude 2 
    [<a href="https://x.com/mattshumer_/status/1699093734330183937" target="_blank">source</a>].
  </p>
</div>





## How Does Speculative Decoding Work?
Speculative decoding is an effective technique developed to accelerate LLMs inference, especially during the decode phase[^speculative_sampling]. It uses a smaller auxiliary model called the draft model. This draft model is compact and fast, enabling it to quickly propose multiple candidate tokens. Although it operates autoregressively, the draft model significantly reduces latency.

Next, the target model (the original model) verifies these candidate tokens in parallel. During this verification, the target model decides whether to accept or reject each draft token. Accepted tokens are immediately appended to the output sequence, significantly speeding up inference. If any tokens are rejected, the target model recomputes the next token sequentially to ensure output quality. By accepting multiple candidate tokens generated by the draft model in a single inference step, the target model effectively reduces the overall latency.

It is crucial to maintain the target model’s performance when candidate tokens from the draft model are accepted. To achieve this, a technique called Speculative Sampling has been proposed. It has been proven that this method theoretically ensures the generated tokens precisely follow the distribution of the target model. Below is the Speculative Sampling algorithm.

---
>$\textbf{Algorithm 1 }$ Speculative Decoding

<div style="font-size: 80%; text-align: left;">
    
$$
\begin{flalign*}
& \textbf{Inputs: } M_p, M_q, \text{ prefix} && \\
& \triangleright \text{Sample } \gamma \text{ guesses } x_1, \ldots, x_\gamma \text{ from } M_q \text{ autoregressively.} && \\
& \textbf{for } i=1 \textbf{ to } \gamma \textbf{ do} && \\
& \quad q_i(x) \leftarrow M_p(\text{prefix} + [x_1, \ldots, x_{i-1}]) && \\
& \quad x_i \sim q_i(x) && \\
& \textbf{end for} && \\
& \triangleright \text{Run } M_p \text{ in parallel.} && \\
& p_1(x), \ldots, p_{\gamma+1}(x) \leftarrow M_p(\text{prefix}), \ldots, M_p(\text{prefix} + [x_1, \ldots, x_\gamma]) && \\
& \triangleright \text{Determine the number of accepted guesses } n && \\
& r_1 \sim U(0,1), \ldots, r_\gamma \sim U(0,1) && \\
& n \leftarrow \min \left( \{ i - 1 \mid 1 \leq i \leq \gamma, r_i > \frac{p_i(x)}{q_i(x)} \} \cup \{\gamma\} \right) && \\
& \triangleright \text{Adjust the distribution from } M_p \text{ if needed.} && \\
& p'(x) \leftarrow p_{n+1}(x) && \\
& \textbf{if } n < \gamma \textbf{ then} && \\
& \quad p'(x) \leftarrow \mathrm{norm}(\max(0, p_{n+1}(x) - q_{n+1}(x))) && \\
& \textbf{end if} && \\
& \triangleright \text{Return one token from } M_p, \text{ and } n \text{ tokens from } M_q. && \\
& t \sim p'(x) && \\
& \textbf{return } \text{prefix} + [x_1, \ldots, x_n, t] && \\
& 1 \leq i \leq \gamma &&
\end{flalign*}
$$
</div>

---
## The Need for Long-Context Speculative Decoding
### Dual Memory Bottlenecks in LLMs: Model Parameters and KV Cache
We found an interesting observation while researching speculative decoding to improve the efficiency of LLM decoding. We conducted experiments under various conditions to measure the speedup using Llama-2-7b-chat as the target model and Llama-68M as the draft model. As we increased the number of generated tokens from 128 to 4096, which is the maximum sequence length that Llama-2-7b-chat can handle, we observed that the effectiveness of speculative decoding diminished as the number of generated tokens grew.


    
<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/S1J-5bLzxl.png" alt="speed_up_context" width="600"/>
</div>
  <div style="text-align: left; font-size: 0.9em; color: gray; margin: 0 auto 20px;">
  Figure 2. Comparison of token generation speeds between speculative decoding and greedy decoding across varying numbers of generated tokens. evaluated on a single RTX-3090 GPU.
</div>
    
The bar chart shows tokens per second for both methods, while the line plot illustrates the speedup ratio (speculative over greedy). The data indicates that speculative decoding provides notable speedup for shorter generation lengths, but its advantage diminishes as the number of generated tokens increases. This decline in effectiveness is mainly caused by the growing cache size. As the cache gets larger, memory use and bandwidth demands increase, adding overhead and reducing the speedup from speculative decoding. Motivated by this issue, we focused on the performance decline of speculative decoding in long context scenarios and explored relevant studies.
<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/rJBUFZIGle.png" alt="model_context_cache" width="600"/>
</div>
  <div style="text-align: left; font-size: 0.9em; color: gray; margin: 0 auto 20px;">
  Figure 3. Memory usage breakdown of KV Cache and model weights across varying context lengths.
</div>
    
## Understanding KV-Cache Behavior for Efficient Long-Context Decoding
### Attention Sparsity
During inference in large language models (LLMs), it has been observed that only a small subset of tokens receive high attention scores, while most tokens receive very low scores. Over 95% of the attention matrix is effectively sparse, which is a well-established finding[^H2O]. In addition, many attention heads tend to focus heavily on specific tokens—a phenomenon known as attention sinks. This behavior is consistently observed across various models and layers, with many heads exhibiting similar sparsity patterns[^AttentionSink].

These observations offer valuable insight into how the KV-Cache can be managed more efficiently. Rather than being a superficial characteristic, attention sparsity presents a meaningful opportunity for optimization: by evicting unimportant KV entries and retaining only the most relevant ones, both memory usage and computational cost can be significantly reduced.
### Leveraging Local Redundancy in KV-Cache Design
The KV-Cache can be divided into smaller chunks, allowing each to be selectively accessed through a retrieval-based approach with little to no impact on performance[^triforce]. By fetching only the relevant information needed for attention computation, this method helps reduce memory usage and avoid unnecessary computation. Although performance may vary depending on chunk size and retrieval strategy, this approach is widely seen as an effective and practical way to support efficient inference.
    
## :deciduous_tree: HiTree: Hierarchical Speculation via Tree-Based Drafting
While exploring ways to improve speculative decoding in long-context scenarios, we came across the [triforce framework](https://github.com/Infini-AI-Lab/TriForce). Although it introduced an efficient cache management strategy, it did not incorporate a tree-based drafting mechanism. This led us to investigate whether integrating a tree structure into the Triformer approach could further enhance performance, which became the starting point of our experiments.
### Hierarchical Speculative Decoding
Triforce is a hierarchical speculative decoding framework designed to improve memory efficiency and computational performance during inference in large language models (LLMs), particularly in long-context scenarios.

The system adopts a hierarchical decoding architecture, where a smaller model first drafts candidate tokens, and the full target model verifies these candidates. One of the core components of Triforce is its KV Retrieval mechanism, which partitions the KV-Cache into smaller chunks and maintains lightweight summary information for each. This design reduces unnecessary memory access, thereby enabling more efficient use of computational and storage resources.
    
### HiTree Design Overview
Tree-based speculative decoding is motivated by the observation that strategically leveraging top-ranked predictions can enable the generation of multiple tokens per decoding step[^medusa][^specinfer][^eagle]. Recent studies have explored similar approaches by incorporating tree-attention mechanisms, and in our work, we focus on integrating a structured tree with a fixed pattern, rather than a dynamically expanding one during inference. The design overview shown in Figure.<a href="#design-overview">4</a>. presents a retrieval-based speculative decoding framework where decoding is performed hierarchically. We introduce a tree-structured expansion of candidate tokens between the smaller draft model and the draft model, aiming to increase the expected length of accepted sequences in each decoding step.
    
    
<a id="design-overview"></a>
<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/HkpTWaLzge.png" alt="efficient_ML_proposal" width="600"/>
</div>
  <div style="text-align: center; font-size: 0.9em; color: gray; margin: 0 auto 20px;">
  Figure 4. Design overview of HiTree
</div>


Figure.<a href="#tree_application">5</a> provides a more detailed view of the hierarchical speculation structure with tree-based drafting. In our design, the tree is applied between the draft and retrieval models. Specifically, the smallest draft model generates a tree of depth $\gamma_1$ using a top-k branching strategy, resulting in $k^{\gamma_1}$ candidate sequences of length $\gamma_1$ .

The retrieval model then selects tokens from these candidates until a total of $\gamma_2$  tokens have been accepted. These selected tokens are finally verified by the target model. This entire process is repeated until the final generated sequence reaches the externally specified length n.

In our setup, we did not apply a tree structure between the retrieval and target models, as the acceptance rate between these two models in the baseline TriForce setup was already sufficiently high.
    
<a id="tree_application"></a>
<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/B165eIvfex.png" alt="efficient_ML_proposal" width="600"/>
</div>
  <div style="text-align: center; font-size: 0.9em; color: gray; margin: 0 auto 20px;">
  Figure 5. Tree architecture adoption between the draft model and retrieval model
</div>


    
## :llama: How Fast Can a Llama Spit?
### Experimental Settings
Our experiments were conducted based on the Triforce framework, as previously described. To ensure a fair comparison with Triforce, we adopted most of the original experimental settings. We used Llama2 and LWM models with a 128K context window as our target models[^LWM].
    
In this setup, we employed a 4K retrieval cache as an intermediate draft cache,
and used the JackFram/Llama68M (JF68M) model (Miao et al., 2023) as the initial draft model. Due to the resource constraints associated with long-context experiments,
we conducted our experiments under an offloading setting. Specifically, we maximized GPU memory utilization by filling it as much as possible, then offloaded the remaining KV cache to the CPU, while keeping the model weights residing on the GPU.

Evaluation was performed on the PG-19 dataset using 100 test examples,
with a target generation length of 256 tokens,
and we measured speedup as the primary performance metric.

    

### Main Results

<a id="overall_result"></a>
<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/rJaWPLDfge.png" alt="overall_result" width="700"/>
</div>
  <div style="text-align: left; font-size: 0.9em; color: gray; margin: 0 auto 20px;">
  Figure 6. Comparison of the acceptance ratio and throughput between the non-tree and tree architecture.
</div>

    
We conducted experiments across various combinations of $\gamma1$ and $\gamma2$.
Specifically, we used $\gamma1$ values of 2, 4, and 6, and $\gamma2$ values of 6 and 12, following the configuration adopted in Triforce. In our setup, the depth of the tree was set equal to $\gamma1$, and we employed a fixed binary tree structure to perform tree-based speculative decoding.
This design allowed us to simplify the implementation while maintaining consistency with the hierarchical decoding framework.
    

We firstly conducted an experiment about the effect of the tree-based architecture in the hierarchy. Figure.<a href="#overall_result">6</a> illustrates the acceptance rate and throughput within the hierarchical structure, specifically between the target model and the retrieval-draft model, as well as between the retrieval model and the draft model.
Since tree-based drafting is applied only between the draft and retrieval models, the behavior between the target and retrieval models remains consistent across both tree-based and non-tree settings. In contrast, between the retrieval and draft models, we observed a significant improvement in both acceptance rate and throughput, highlighting the effectiveness of the tree-based drafting strategy in this stage.
    
As shown in Figure.<a href="#speed_up_hitree">7</a>, under the configuration of γ₁ = 4 and γ₂ = 12, HiTree achieved a 17.7% speedup over Triforce and demonstrated 1.72× faster performance than the autoregressive (AR) baseline.
In contrast, in the setting where γ₁ = 4 and γ₂ = 6, where Triforce's speedup was relatively strong, HiTree’s advantage was less evident.
  
Across all experimental settings, we observed that HiTree, which employs tree-based hierarchical speculative decoding, consistently achieved higher speedup compared to Triforce.
    
    
<a id="speed_up_hitree"></a>
<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/rymQiSDGll.png" alt="speed_up_hitree" width="700"/>
</div>
  <div style="text-align: left; font-size: 0.9em; color: gray; margin: 0 auto 20px;">
  Figure 7. Comparison of speedup between Triforce and HiTree under various settings.
A speedup value of 1 corresponds to the decoding speed of the autoregressive (AR) baseline.
</div>
Additionally, HiTree, which uses a binary tree structure to explore candidate tokens, achieved a higher acceptance rate than Triforce.
While the acceptance rate improved by up to nearly 2×, the corresponding speedup gain was less than expected, likely due to the overhead introduced by the tree structure. 
    
As shown in Figure.<a href="#overall_result">6</a>, we observed a significant drop in the acceptance rate for Triforce as γ increased. This is perhaps unsurprising, as Triforce performs autoregressive decoding based on tokens generated by a smaller draft model.
    
In contrast, HiTree maintained a more stable acceptance rate even as γ₁ increased, suggesting that the tree structure provides greater robustness during speculative verification.
    


<a id="acceptance"></a>
<div style="text-align: center;">
  <img src="https://hackmd.io/_uploads/rJCbZUvzee.png" alt="acceptance" width="700"/>
</div>
  <div style="text-align: center; font-size: 0.9em; color: gray; margin: 0 auto 20px;">
  Figure 8. Ablation for tree size that depends on the number of branches
</div>

We additionally conducted an ablation study by varying the number of branches per node to analyze the effect of different tree structures on performance.

Thanks to the parallel nature of tree construction, we observed minimal increase in generation latency across different branching factors. Notably, setting the branch size to 3 resulted in a meaningful improvement in acceptance rate. This indicates that, within a reasonable memory budget, increasing tree size can be an effective strategy.

However, the memory required for KV cache grows exponentially with the number of branches, while the corresponding gains in acceptance rate diminish. As a result, increasing the branch factor beyond 4 is likely to be inefficient due to the unfavorable trade-off between memory usage and performance improvement.

### Limitation & Future Work
In this study, we conducted our experiments using a fixed binary tree structure.
This approach offers the advantage of simplicity in implementation and clarity in inference flow design.
However, it also has limitations, as it does not guarantee optimal candidate exploration across diverse scenarios.

Future work could extend the HiTree framework by dynamically constructing the tree and adapting its structure based on token distributions.
Such an approach would enable more flexible and efficient candidate branching tailored to the context, and has the potential to further maximize acceptance rate and reduce latency.

    
## Conclusion
In this project, we propose HiTree, a novel LLM decoding framework that integrates hierarchical speculative decoding with tree-based drafting. We observed that existing speculative decoding methods often struggle with long-context generation. By incorporating a tree-based drafting strategy between the draft and target models, HiTree achieves up to 3.23× higher acceptance rates compared to using hierarchical speculation alone, ultimately leading to a 17% reduction in latency.



[^splitwise]: Patel, Pratyush, et al. "Splitwise: Efficient generative llm inference using phase splitting." 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA). IEEE, 2024. [[pdf]](https://arxiv.org/pdf/2311.18677)
[^speculative_sampling]: Leviathan, Yaniv, Matan Kalman, and Yossi Matias. "Fast inference from transformers via speculative decoding." International Conference on Machine Learning. PMLR, 2023. [[pdf]](https://arxiv.org/pdf/2211.17192)
[^H2O]: Zhang, Zhenyu, et al. "H2o: Heavy-hitter oracle for efficient generative inference of large language models." Advances in Neural Information Processing Systems 36 (2023): 34661-34710. [[pdf]](https://arxiv.org/pdf/2306.14048)
[^AttentionSink]: Xiao, Guangxuan, et al. "Efficient streaming language models with attention sinks." arXiv preprint arXiv:2309.17453 (2023). [[pdf]](https://arxiv.org/pdf/2309.17453)
[^triforce]: Sun, Hanshi, et al. "Triforce: Lossless acceleration of long sequence generation with hierarchical speculative decoding." arXiv preprint arXiv:2404.11912 (2024). [[pdf]](https://arxiv.org/pdf/2404.11912)
[^medusa]: Cai, Tianle, et al. "Medusa: Simple llm inference acceleration framework with multiple decoding heads." arXiv preprint arXiv:2401.10774 (2024). [[pdf]](https://arxiv.org/pdf/2401.10774)
[^specinfer]: Miao, Xupeng, et al. "Specinfer: Accelerating large language model serving with tree-based speculative inference and verification." Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3. 2024. [[pdf]](https://arxiv.org/pdf/2305.09781)
[^eagle]: Li, Yuhui, et al. "Eagle: Speculative sampling requires rethinking feature uncertainty." arXiv preprint arXiv:2401.15077 (2024). [[pdf]](https://arxiv.org/pdf/2401.15077)
[^LWM]: Peng, Bowen, et al. "Yarn: Efficient context window extension of large language models." arXiv preprint arXiv:2309.00071 (2023). [[pdf]](https://arxiv.org/pdf/2309.00071)
