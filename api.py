from flask import Flask, jsonify, request
from datetime import datetime

app = Flask(__name__)

usuarios = [
    {"id": 1, "nome": "João Silva", "email": "joao@email.com", "criado_em": "2024-01-01"},
    {"id": 2, "nome": "Maria Santos", "email": "maria@email.com", "criado_em": "2024-01-02"}
]
proximo_id = 3

def encontrar_usuario(id):
    return next((u for u in usuarios if u["id"] == id), None)

@app.route('/')
def home():
    return jsonify({
        "mensagem": "API de Usuários",
        "versao": "1.0",
        "endpoints": {
            "GET /api/usuarios": "Listar todos os usuários",
            "GET /api/usuarios/<id>": "Buscar usuário por ID", 
            "POST /api/usuarios": "Criar novo usuário",
            "PUT /api/usuarios/<id>": "Atualizar usuário",
            "DELETE /api/usuarios/<id>": "Deletar usuário"
        }
    })

@app.route('/api/usuarios', methods=['GET'])
def listar_usuarios():
    return jsonify({
        "usuarios": usuarios,
        "total": len(usuarios)
    })

@app.route('/api/usuarios/<int:id>', methods=['GET'])
def buscar_usuario(id):
    usuario = encontrar_usuario(id)
    if usuario:
        return jsonify(usuario)
    return jsonify({"erro": "Usuário não encontrado"}), 404

@app.route('/api/usuarios', methods=['POST'])
def criar_usuario():
    global proximo_id
    
    dados = request.get_json()

    if not dados or 'nome' not in dados or 'email' not in dados:
        return jsonify({"erro": "Nome e email são obrigatórios"}), 400

    email_existe = any(u["email"] == dados["email"] for u in usuarios)
    if email_existe:
        return jsonify({"erro": "Email já cadastrado"}), 400
    
    novo_usuario = {
        "id": proximo_id,
        "nome": dados["nome"],
        "email": dados["email"],
        "criado_em": datetime.now().strftime("%Y-%m-%d")
    }
    
    usuarios.append(novo_usuario)
    proximo_id += 1
    
    return jsonify(novo_usuario), 201

@app.route('/api/usuarios/<int:id>', methods=['PUT'])
def atualizar_usuario(id):
    usuario = encontrar_usuario(id)
    if not usuario:
        return jsonify({"erro": "Usuário não encontrado"}), 404
    
    dados = request.get_json()
    if not dados:
        return jsonify({"erro": "Dados não fornecidos"}), 400

    if 'nome' in dados:
        usuario['nome'] = dados['nome']
    if 'email' in dados:
        email_existe = any(u["email"] == dados["email"] and u["id"] != id for u in usuarios)
        if email_existe:
            return jsonify({"erro": "Email já está em uso"}), 400
        usuario['email'] = dados['email']
    
    return jsonify(usuario)

@app.route('/api/usuarios/<int:id>', methods=['DELETE'])
def deletar_usuario(id):
    global usuarios
    usuario = encontrar_usuario(id)
    if not usuario:
        return jsonify({"erro": "Usuário não encontrado"}), 404
    
    usuarios = [u for u in usuarios if u["id"] != id]
    return jsonify({"mensagem": "Usuário deletado com sucesso"})

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"erro": "Requisição inválida"}), 400

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"erro": "Método não permitido"}), 405

if __name__ == '__main__':
    print("🚀 Iniciando API de Usuários...")
    print("📍 Acesse: http://localhost:5000")
    print("📋 Documentação: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

