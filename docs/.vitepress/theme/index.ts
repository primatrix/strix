import DefaultTheme from 'vitepress/theme'
import giscusTalk from 'vitepress-plugin-comment-with-giscus'
import { useData, useRoute } from 'vitepress'

export default {
  ...DefaultTheme,
  setup() {
    const { frontmatter } = useData()
    const route = useRoute()

    giscusTalk(
      {
        repo: 'primatrix/wiki',
        repoId: 'R_kgDORtfQfg',
        category: 'Comments',
        categoryId: 'DIC_kwDORtfQfs4C6ut1',
        mapping: 'pathname',
        inputPosition: 'top',
        lang: 'zh-CN',
        homePageShowComment: false,
        lightTheme: 'light',
        darkTheme: 'transparent_dark',
      },
      {
        frontmatter,
        route,
      },
      true,
    )
  },
}
