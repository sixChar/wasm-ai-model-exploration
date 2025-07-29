package main

import (
    "net/http"
    "html/template"
    "os"
    "log"
)


type TemplateData struct {
    BasePath string
}


func main() {
    if len(os.Args) != 2 {
        log.Println(`
            USAGE:
                go run main.go PORT
        `)
        os.Exit(1)
    }
    port := os.Args[1]
    
    fs := http.FileServer(http.Dir("./static"))
    http.Handle("/static/", http.StripPrefix("/static/", fs))

    tmpl := template.Must(template.ParseFiles("templates/index.html"))
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        basePath := r.Header.Get("SCRIPT_NAME")

        if basePath == "/" {
            basePath = ""
        }

        data := TemplateData{
            BasePath: basePath,
        }

        if err := tmpl.Execute(w, data); err != nil {
            http.Error(w, err.Error(), http.StatusInternalServerError)
        }
    })


    log.Println("Listening on port " + port)
    if err := http.ListenAndServe(":" + port, nil); err != nil {
        log.Fatal(err)
    }
    
}
