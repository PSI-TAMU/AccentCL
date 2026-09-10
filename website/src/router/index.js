import { createRouter, createWebHistory } from 'vue-router'
import PaperView from '../PaperView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: PaperView
    }
  ],
})

export default router
