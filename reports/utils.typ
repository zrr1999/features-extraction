#let include-section(title) = {
  // heading(level: 1)[#{ title } 复现情况]
  include "sections/" + title + ".typ"
  pagebreak()
}
