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

  set par(spacing: 1.5em)

  let number-to-letter(n) = {
    str.from-unicode(96+n)
  }

  set heading(
    numbering: (..nums) => {
      let level = nums.pos().len()
      let val = nums.pos().last()
      if level == 1 { return str(val) + "." }
      else if level == 2 { return number-to-letter(val) + "." }
      else if level == 3 { return str(val) + ")" }
    }
  )

  show heading.where(level: 1): it => {
    counter(math.equation).update(0)
    counter(figure).update(0)
    it
  }
  
  set math.equation(
    numbering: (..nums) => {
      context {
        let chapter = counter(heading).get().first()
        let eq_num = nums.pos().first()
        return "(" + str(chapter) + "." + str(eq_num) + ")"
      }
    }
  )

  set figure(
    numbering: (..nums) => {
      context {
        let chapter = counter(heading).get().first()
        let fig_num = nums.pos().first()
        return str(chapter) + "." + str(fig_num)
      }
    }
  )

  show figure.where(kind: image): set figure(supplement: "Gambar")
  show figure.where(kind: table): set figure(supplement: "Tabel")
  
  show figure.caption: it => [
    #emph[ #it.supplement #it.counter.display(it.numbering): #it.body ]
  ]

  set page(numbering: none)
  align(center)[
    #v(2cm)
    #text(size: 14pt)[PROPOSAL SKRIPSI]
    #v(2cm)
    #text(size: 14pt)[#title]
    #v(3cm)
    Diajukan oleh: \
    #v(1cm)
    #author \
    #npm \
    #email
    #v(3cm)
    Dosen Pembimbing Proposal Skripsi: \
    #v(1cm)
    #supervisor
    #v(1cm)
    #v(1fr)
    #program \
    #dept \
    #faculty \
    #date \
  ]
  pagebreak()
  
  set page(numbering: "1")
  counter(page).update(1)
  body
}
