import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    lang: 'zh-CN',
    title: 'Internal Wiki',
    description: '内部技术知识库',

    lastUpdated: true,
    ignoreDeadLinks: true,

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
            { text: 'Project Alpha', link: '/projects/project-alpha/' },
            { text: 'Project Beta', link: '/projects/project-beta/' },
            { text: 'Ling 对齐', link: '/projects/ling-alignment/' },
            { text: '性能优化', link: '/projects/performance-optimization/' },
          ],
        },
        { text: 'RFC', link: '/rfc/' },
        { text: '最佳实践', link: '/best-practices/' },
        { text: '新人入门', link: '/onboarding/' },
      ],

      // Sidebar (manual configuration per section)
      sidebar: {
        '/projects/project-alpha/': [
          {
            text: 'Project Alpha',
            items: [
              { text: '概览', link: '/projects/project-alpha/' },
              {
                text: '教程指南',
                collapsed: false,
                items: [
                  { text: '快速开始', link: '/projects/project-alpha/guide/getting-started' },
                ],
              },
              {
                text: '技术参考',
                collapsed: true,
                items: [
                  { text: 'API 文档', link: '/projects/project-alpha/reference/api' },
                ],
              },
              {
                text: '故障排查',
                collapsed: true,
                items: [
                  { text: '概览', link: '/projects/project-alpha/troubleshooting/' },
                ],
              },
            ],
          },
        ],
        '/projects/project-beta/': [
          {
            text: 'Project Beta',
            items: [
              { text: '概览', link: '/projects/project-beta/' },
              {
                text: '教程指南',
                collapsed: false,
                items: [
                  { text: '概览', link: '/projects/project-beta/guide/' },
                ],
              },
              {
                text: '技术参考',
                collapsed: true,
                items: [
                  { text: '概览', link: '/projects/project-beta/reference/' },
                ],
              },
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
