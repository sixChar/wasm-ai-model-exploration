package main

import (
    "fmt"
    "net/http"
    "html/template"
)




func main() {
    
    fs := http.FileServer(http.Dir("./static"))
    http.Handle("/static/", http.StripPrefix("/static/", fs))

    tmpl := template.Must(template.ParseFiles("templates/index.html"))
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        tmpl.Execute(w, nil)
    })


    fmt.Println("Listening on port 8086...")
    http.ListenAndServe(":8086", nil)
    
    
}
