#let project(
  title: "",
  author: "",
  npm: "",
  email: "",
  supervisor: "",
  dept: "",
  faculty: "",
  program: "",
  date: "",
  body
) = {
  set document(author: author, title: title)
  set page(
    paper: "a4",
    margin: (left: 4cm, top: 3cm, right: 3cm, bottom: 3cm),
    numbering: "1"
  )
  set text(font: "Times New Roman", size: 12pt, lang: "id")

  set par(
    justify: true,
    leading: 1.5em,
    first-line-indent: 1cm,
  )

  show par: set block(spacing: 1.5em)

  let number-to-letter(n) = {
    str.from-unicode(96+n)
  }

  set heading(
    numbering: (..nums) => {
      let level = nums.pos.len()
      let val = nums.pos.last()
      if level == 1 { return str(val) + "." }
      else if level == 2 { return number-to-letter(val) + "." }
      else if level == 3 { return str(val) + ")" }
    }
  )

  show heading.where(level: 1): it => {
    counter(math.equation).update(0)
    it
  }
  
  set math.equation(
    numbering: (..nums) => {
      context {
        let chapter = counter(heading).get().first()
        let eq_num = nums.pos.first()
        return "(" + str(chapter) + "." + str(eq_num) + ")"
      }
    }
  )
    
  
}
