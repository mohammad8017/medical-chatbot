import customtkinter as ctk
import tkinter.messagebox as tkmb
import threading
import time
import random
from send_result import main


models = ["Bert", "tf-idf", "gpt-2"]

# Selecting GUI theme - dark, light , system (for system default)
ctk.set_appearance_mode("dark")

# Selecting color theme - blue, green, dark-blue
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("700x850")
app.title("Chatbot")



def submit():

	selected_model = model_var.get()
	user_question = question_text.get()

	if selected_model in models:
		question_result_text.configure(state="normal")
		question_result_text.delete("1.0", "end")
		user_question_tmp = user_question.split(' ')
		user_question_tmp.reverse()
		question_result_text.insert("1.0", ' '.join(user_question_tmp))
		question_result_text.configure(state="disabled")

		answer_result_text.configure(state="normal")
		answer_result_text.delete("1.0", "end")
		answer_result_text.insert("1.0", "Generating answer...")
		answer_result_text.configure(state="disabled")

		thread = threading.Thread(target=calculate_answer, args=(selected_model, user_question))
		thread.start()


# Function to calculate the answer
def calculate_answer(selected_model, user_question):
	# Simulate a time-consuming calculation (replace this with your actual computation)
	# time.sleep(10)  # This can take more or less than 10 seconds
	# answers = ["ans1", "ans2", "ans3", "ans4", "ans5"]
	# answer = random.choice(answers)
	answer = main(user_question, selected_model)

	# Update the GUI with the answer
	answer_result_text.configure(state="normal")
	answer_result_text.delete("1.0", "end")
	answer_result_text.insert("1.0", answer)
	answer_result_text.configure(state="disabled")


label = ctk.CTkLabel(app,text="Chatbot")

# label.pack(pady=20


frame = ctk.CTkFrame(master=app)
frame.pack(pady=20,padx=40,fill='both',expand=True)


# label = ctk.CTkLabel(master=frame,text='Modern Login System UI')
# label.pack(pady=12,padx=10)


# model_var= ctk.CTkEntry(master=frame,placeholder_text="Model")
# model_var.pack(pady=12,padx=10)
label = ctk.CTkLabel(master=frame, text="Question answering chatbot using bert and tfidf to answer your questions\n\n")
label.pack(pady=2,padx=10)

model_var = ctk.CTkComboBox(master=frame, values=['Bert', "tf-idf", "gpt-2"])
model_var.pack(pady=12,padx=10)
model_var.set("gpt-2")

question_text= ctk.CTkEntry(master=frame,placeholder_text="Question", height=100, width=500)
question_text.pack(pady=12,padx=10)


button = ctk.CTkButton(master=frame,text='Submit',command=submit)
button.pack(pady=12,padx=10)

# checkbox = ctk.CTkCheckBox(master=frame,text='Remember Me')
# checkbox.pack(pady=12,padx=10)


question_label = ctk.CTkLabel(master=frame, text="question:")
question_label.pack(pady=2,padx=10)

question_result_text = ctk.CTkTextbox(master=frame, width=500, height=150)
question_result_text.pack(pady=12,padx=10)
question_result_text.configure(state="disabled")


answer_label = ctk.CTkLabel(master=frame, text="answer:")
answer_label.pack(pady=2,padx=10)

answer_result_text = ctk.CTkTextbox(master=frame, width=500, height=150)
answer_result_text.pack(pady=12,padx=10)
answer_result_text.configure(state="disabled")


app.mainloop()
