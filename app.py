from shiny import App,render,ui

ui_=ui.page_fluid(ui.input_slider("slider0","slider0 label",10,100,10),ui.output_text_verbatim("txt"),
    title="app.py")

def server(input,output,session):
    @output
    @render.text
    def txt():
        return f"number is {input.slider0()}"


app=App(ui_,server)