# Usamos framework para interfaz
import streamlit as st
# Importamos el script e importar funciones que usaremos
from chatbot import predict_class, get_response, intents
# Título de la app
st.title("💻 👨‍💻 👩‍💻 Asistente virtual U catolica")


# Definimos variables de estado para almacenar historico del chat y otra para ver si es la primera vez que el user entra al chat
if "messages" not in st.session_state:
    st.session_state.messages = []  # Este array almacenará el historial del chat
if "first_message" not in st.session_state:
    # Esta variable es para saber si es la 1 vez en usar el chat
    # y cuando mostremos el primer mensaje se cambiará a FALSE
    st.session_state.first_message = True

# Recorremos nuestro array de mensajes para ver si tenemos algo en el historial del chat
for message in st.session_state.messages:
    # Usamos el método chat_message para decirle el rol que puede ser el del chatbot o el user
    with st.chat_message(message["role"]):
        # Aca mostramos el mensaje del historial
        st.markdown(message["content"])

# Comprobar si es la primera vez que el usuario ejecuta el código, y en caso de ser así mostrar la bienvenida al usuario
# Con este condicional comprobamos si es la primera vez que se ejecuta el código y en caso de ser así
# con las mismas dos sentencias pues le mostramos el mensaje de bienvenida
if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("Hola soy el ChatBot Ucatolico, ¿cómo puedo ayudarte?")
    # Aca añadimos los mensajes al historial con el append
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hola, ¿cómo puedo ayudarte?"}
    )
    st.session_state.first_message = False

if prompt := st.chat_input("¿Cómo puedo ayudarte?"):
    # 'st.chat_input()' muestra un cuadro de texto en la interfaz.
    # Si el usuario escribe algo, se guarda en la variable 'prompt'.
    # El operador ':=' (walrus operator) asigna y evalúa al mismo tiempo.
    # Ejemplo: prompt = "Hola", entonces la condición es True y entra al bloque.

    with st.chat_message("user"):
        # Abre un bloque de mensaje en la interfaz de chat de Streamlit con el rol "usuario".
        st.markdown(prompt)
        # Muestra en pantalla el texto que el usuario escribió.

        st.session_state.messages.append({"role": "user", "content": prompt})
        # Guarda el mensaje del usuario en la memoria de la sesión (para historial).
        # Esto permite que la conversación no se pierda al recargar la página.

    # ------------------- IA -------------------
    # Implementación del algoritmo de IA
    insts = predict_class(prompt)
    # Llama a la función 'predict_class', que toma el texto del usuario (prompt),
    # lo convierte a bolsa de palabras (bag of words), y usa el modelo para
    # predecir la intención (intent) más probable. Devuelve una lista con el intent.

    res = get_response(insts, intents)
    # Con la intención predicha, se busca en el JSON de intents
    # y se elige aleatoriamente una respuesta asociada a ese intent.

    # ------------------- BOT -------------------
    with st.chat_message("assistant"):
        # Abre un bloque de mensaje con el rol "asistente" (el chatbot).
        st.markdown(res)
        # Muestra en pantalla la respuesta que el bot generó.

        st.session_state.messages.append({"role": "assistant", "content": res})
        # Guarda la respuesta del bot en el historial de sesión.
