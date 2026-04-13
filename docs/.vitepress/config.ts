import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    lang: 'zh-CN',
    title: 'Internal Wiki',
    description: '内部技术知识库',

    lastUpdated: true,
    ignoreDeadLinks: true,
    srcExclude: ['superpowers/**'],

    markdown: {
      math: true,
    },

    themeConfig: {
      // Chinese UI localization
      lastUpdatedText: '最后更新',
      returnToTopLabel: '返回顶部',
      sidebarMenuLabel: '菜单',
      darkModeSwitchLabel: '主题',
      outlineTitle: '本页目录',
      docFooter: { prev: '上一页', next: '下一页' },

      // Navigation bar
      nav: [
        { text: '首页', link: '/' },
        {
          text: '项目',
          items: [
            { text: 'Beaver', link: '/projects/beaver/' },
            { text: 'sglang-jax', link: '/projects/sglang-jax/' },
            { text: 'Ling 对齐', link: '/projects/ling-alignment/' },
            { text: '性能优化', link: '/projects/performance-optimization/' },
            { text: 'pallas-kernel', link: '/projects/pallas-kernel/' },
            { text: 'Falcon', link: '/projects/falcon/' },
          ],
        },
        { text: 'RFC', link: '/rfc/' },
        { text: '最佳实践', link: '/best-practices/' },
        { text: '新人入门', link: '/onboarding/' },
      ],

      // Sidebar (manual configuration per section)
      sidebar: {
        '/projects/beaver/': [
          {
            text: 'Beaver',
            items: [
              { text: '概览', link: '/projects/beaver/' },
              { text: '产品设计', link: '/projects/beaver/design' },
              { text: '标签体系规范', link: '/projects/beaver/label-system' },
              { text: '通知机制规范', link: '/projects/beaver/notification-system' },
              { text: 'GitHub 集成规范', link: '/projects/beaver/github-integration' },
              { text: 'Cloudflare 基础架构设计', link: '/projects/beaver/cloudflare-infrastructure' },
            ],
          },
        ],
        '/projects/ling-alignment/': [
          {
            text: 'Ling 对齐',
            items: [
              { text: '概览', link: '/projects/ling-alignment/' },
              {
                text: '教程指南',
                collapsed: false,
                items: [
                  { text: 'Megatron-LM Dump 指南', link: '/projects/ling-alignment/guide/megatron-dump' },
                ],
              },
            ],
          },
        ],
        '/projects/sglang-jax/': [
          {
            text: 'sglang-jax',
            items: [
              { text: '概览', link: '/projects/sglang-jax/' },
              { text: '文档设计', link: '/projects/sglang-jax/2026-03-26-project-documentation-design' },
              {
                text: '参考资料',
                collapsed: false,
                items: [
                  {
                    text: 'tpu-inference Q2 对照',
                    link: '/projects/sglang-jax/reference/tpu-inference-q2',
                    collapsed: false,
                    items: [
                      { text: 'MLA 与 DeepSeek V3', link: '/projects/sglang-jax/reference/gap-mla-deepseek' },
                      { text: 'DP Attention', link: '/projects/sglang-jax/reference/gap-dp-attention' },
                      { text: 'PD Disaggregation', link: '/projects/sglang-jax/reference/gap-pd-disaggregation' },
                      { text: 'GMM Kernel', link: '/projects/sglang-jax/reference/gap-gmm-kernel' },
                      { text: '通信-计算重叠', link: '/projects/sglang-jax/reference/gap-allgather-matmul' },
                      { text: 'N:M 结构化稀疏', link: '/projects/sglang-jax/reference/gap-sparse-matmul' },
                      { text: '高级优化', link: '/projects/sglang-jax/reference/gap-optimizations' },
                    ],
                  },
                ],
              },
            ],
          },
        ],
        '/projects/performance-optimization/': [
          {
            text: '性能优化',
            items: [
              { text: '概览', link: '/projects/performance-optimization/' },
              { text: '分析框架', link: '/projects/performance-optimization/analysis-framework' },
              { text: '分析结果', link: '/projects/performance-optimization/analysis-results' },
              { text: '工作拆解', link: '/projects/performance-optimization/work-breakdown' },
            ],
          },
        ],
        '/projects/pallas-kernel/': [
          {
            text: 'pallas-kernel',
            items: [
              { text: '项目概述', link: '/projects/pallas-kernel/' },
              { text: '硬件约束与 API 限制', link: '/projects/pallas-kernel/hardware-constraints' },
              { text: '代码规范', link: '/projects/pallas-kernel/coding-standards' },
              { text: 'CI/CD 流水线', link: '/projects/pallas-kernel/ci-cd' },
              { text: '开发流程', link: '/projects/pallas-kernel/development-workflow' },
              { text: 'Benchmark 规范', link: '/projects/pallas-kernel/benchmarking' },
              { text: 'Kernel 文档模板', link: '/projects/pallas-kernel/kernel-template' },
            ],
          },
          {
            text: '性能工程参考',
            collapsed: false,
            items: [
              { text: 'Roofline 分析深度指南', link: '/projects/pallas-kernel/roofline-analysis' },
              { text: 'TPU 硬件规格参考', link: '/projects/pallas-kernel/tpu-specs' },
              { text: 'Sharding 与集合通信', link: '/projects/pallas-kernel/sharding-collectives' },
              { text: 'Transformer 算子性能', link: '/projects/pallas-kernel/transformer-ops-reference' },
              { text: 'Profiling 深度指南', link: '/projects/pallas-kernel/profiling-guide' },
            ],
          },
        ],
        '/projects/falcon/': [
          {
            text: 'Falcon',
            items: [
              { text: '概览', link: '/projects/falcon/' },
              { text: '整体架构与模块分界', link: '/projects/falcon/architecture' },
              { text: '集群编排层设计', link: '/projects/falcon/cluster-layer-design' },
              { text: 'Phase 1 架构设计', link: '/projects/falcon/phase1-design' },
              { text: 'Phase 2 架构设计', link: '/projects/falcon/phase2-design' },
              { text: 'Phase 3 架构设计', link: '/projects/falcon/phase3-design' },
            ],
          },
        ],
        '/rfc/': [
          {
            text: 'RFC',
            items: [
              { text: '概览', link: '/rfc/' },
              { text: 'RFC 模板', link: '/rfc/template' },
              { text: 'RFC-0001: 示例', link: '/rfc/0001-example-rfc' },
            ],
          },
        ],
        '/best-practices/': [
          {
            text: '最佳实践',
            items: [
              { text: '概览', link: '/best-practices/' },
              { text: '代码审查', link: '/best-practices/code-review' },
              { text: '编码规范', link: '/best-practices/coding-standards' },
            ],
          },
        ],
        '/onboarding/': [
          {
            text: '新人入门',
            items: [
              { text: '概览', link: '/onboarding/' },
              { text: '开发环境搭建', link: '/onboarding/dev-environment' },
              { text: 'Cloudflare 配置', link: '/onboarding/cloudflare-setup' },
            ],
          },
        ],
      },

      // Edit link
      editLink: {
        pattern: 'https://github.com/primatrix/wiki/edit/main/docs/:path',
        text: '在 GitHub 上编辑此页',
      },

      // Local search with Chinese tokenization
      search: {
        provider: 'local',
        options: {
          translations: {
            button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
            modal: {
              noResultsText: '未找到相关结果',
              resetButtonTitle: '清除',
              footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
            },
          },
          miniSearch: {
            options: {
              tokenize: (text) => {
                const segmenter = new Intl.Segmenter('zh-CN', { granularity: 'word' })
                const segments = [...segmenter.segment(text)]
                return segments
                  .filter((s) => s.isWordLike)
                  .map((s) => s.segment)
              },
            },
            searchOptions: {
              combineWith: 'AND',
              fuzzy: 0.2,
            },
          },
        },
      },

      // Social links
      socialLinks: [
        { icon: 'github', link: 'https://github.com/primatrix/wiki' },
      ],
    },

    // Mermaid plugin configuration
    mermaid: {},
  })
)
