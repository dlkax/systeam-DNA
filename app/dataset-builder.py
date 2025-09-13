"""
🦴 DATASET BUILDER - Versão Robusta
Múltiplas fontes + sistema de fallback
"""

import os
import requests
import time
import hashlib
import json
import shutil
from PIL import Image
from urllib.parse import quote
import random

class RobustDatasetBuilder:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.stats = {"total_tentativas": 0, "sucessos": 0, "erros": 0}
        
        # Múltiplas fontes de imagens
        self.fontes = [
            self.buscar_wikimedia,
            self.buscar_naturalis,
            self.buscar_phylopic,
            self.buscar_gbif,
            self.buscar_eol
        ]

    def buscar_wikimedia(self, termo, limite=15):
        """Wikimedia Commons - Principal fonte científica"""
        print(f"🔬 Wikimedia Commons: {termo}")
        
        url = "https://commons.wikimedia.org/w/api.php"
        
        # Múltiplas variações de busca
        termos_variados = [
            f'{termo} fossil',
            f'{termo} skeleton', 
            f'{termo} museum',
            f'{termo} paleontology',
            f'{termo.replace("_", " ")} bones'
        ]
        
        urls = []
        for termo_var in termos_variados:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': termo_var,
                'srnamespace': 6,
                'srlimit': limite//len(termos_variados)
            }
            
            try:
                response = self.session.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for result in data.get('query', {}).get('search', []):
                        title = result['title'].replace(' ', '_')
                        img_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{title}?width=800"
                        urls.append(img_url)
            except Exception as e:
                print(f"❌ Erro Wikimedia variação '{termo_var}': {e}")
        
        return urls[:limite]

    def buscar_naturalis(self, termo, limite=10):
        """Naturalis Biodiversity Center API"""
        print(f"🏛️  Naturalis Museum: {termo}")
        
        url = "https://api.biodiversitydata.nl/v2/specimen/query"
        
        params = {
            'scientificName': termo.replace('_', ' '),
            'hasMediaObject': True,
            'size': limite
        }
        
        urls = []
        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('resultSet', []):
                    if 'associatedMultiMediaUris' in item:
                        for media in item['associatedMultiMediaUris']:
                            if 'accessUri' in media:
                                urls.append(media['accessUri'])
        except Exception as e:
            print(f"❌ Erro Naturalis: {e}")
        
        return urls

    def buscar_phylopic(self, termo, limite=5):
        """PhyloPic - Silhuetas científicas"""
        print(f"🐾 PhyloPic: {termo}")
        
        # Mapeamento para nomes científicos corretos
        nomes_phylopic = {
            "mammuthus_primigenius": "Mammuthus",
            "smilodon_fatalis": "Smilodon",
            "preguica_gigante": "Megatherium"
        }
        
        nome_busca = nomes_phylopic.get(termo, termo.replace('_', ' '))
        
        try:
            # Buscar UUID do táxon
            search_url = f"https://api.phylopic.org/nodes?filter_name={quote(nome_busca)}"
            response = self.session.get(search_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                urls = []
                
                for node in data.get('_embedded', {}).get('nodes', []):
                    uuid = node.get('uuid')
                    if uuid:
                        # Buscar imagens do node
                        images_url = f"https://api.phylopic.org/nodes/{uuid}/images"
                        img_response = self.session.get(images_url, timeout=10)
                        
                        if img_response.status_code == 200:
                            img_data = img_response.json()
                            for img in img_data.get('_embedded', {}).get('images', []):
                                img_uuid = img.get('uuid')
                                if img_uuid:
                                    img_url = f"https://images.phylopic.org/images/{img_uuid}/vector.svg"
                                    urls.append(img_url)
                                    if len(urls) >= limite:
                                        break
                    if len(urls) >= limite:
                        break
                
                return urls
                
        except Exception as e:
            print(f"❌ Erro PhyloPic: {e}")
        
        return []

    def buscar_gbif(self, termo, limite=10):
        """GBIF - Global Biodiversity Information Facility"""
        print(f"🌍 GBIF: {termo}")
        
        try:
            # Primeiro, buscar o taxonKey
            species_url = "https://api.gbif.org/v1/species/match"
            params = {'name': termo.replace('_', ' ')}
            
            response = self.session.get(species_url, params=params, timeout=10)
            if response.status_code == 200:
                species_data = response.json()
                taxon_key = species_data.get('usageKey')
                
                if taxon_key:
                    # Buscar ocorrências com imagens
                    occurrence_url = "https://api.gbif.org/v1/occurrence/search"
                    params = {
                        'taxonKey': taxon_key,
                        'mediaType': 'StillImage',
                        'limit': limite
                    }
                    
                    occ_response = self.session.get(occurrence_url, params=params, timeout=10)
                    if occ_response.status_code == 200:
                        occ_data = occ_response.json()
                        urls = []
                        
                        for result in occ_data.get('results', []):
                            if 'media' in result:
                                for media in result['media']:
                                    if media.get('type') == 'StillImage':
                                        urls.append(media.get('identifier'))
                        
                        return urls
        except Exception as e:
            print(f"❌ Erro GBIF: {e}")
        
        return []

    def buscar_eol(self, termo, limite=8):
        """Encyclopedia of Life"""
        print(f"📚 EOL: {termo}")
        
        try:
            # Buscar página da espécie
            search_url = "https://eol.org/api/search/1.0.json"
            params = {
                'q': termo.replace('_', ' '),
                'page': 1,
                'exact': True
            }
            
            response = self.session.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                urls = []
                
                for result in data.get('results', []):
                    page_id = result.get('id')
                    if page_id:
                        # Buscar imagens da página
                        page_url = f"https://eol.org/api/pages/1.0/{page_id}.json"
                        page_params = {'images': 75, 'videos': 0, 'sounds': 0, 'maps': 0, 'text': 0}
                        
                        page_response = self.session.get(page_url, params=page_params, timeout=10)
                        if page_response.status_code == 200:
                            page_data = page_response.json()
                            
                            for obj in page_data.get('dataObjects', []):
                                if obj.get('dataType') == 'http://purl.org/dc/dcmitype/StillImage':
                                    img_url = obj.get('mediaURL')
                                    if img_url:
                                        urls.append(img_url)
                                        if len(urls) >= limite:
                                            break
                    if len(urls) >= limite:
                        break
                
                return urls
                
        except Exception as e:
            print(f"❌ Erro EOL: {e}")
        
        return []

    def coletar_urls_todas_fontes(self, especie, termos_busca):
        """Coleta URLs de todas as fontes disponíveis"""
        print(f"\n🔍 COLETANDO URLs PARA: {especie}")
        print("-" * 40)
        
        todas_urls = []
        
        # Para cada termo de busca
        for termo in termos_busca:
            print(f"\n📝 Termo: '{termo}'")
            
            # Tentar cada fonte
            for fonte in self.fontes:
                try:
                    urls = fonte(termo)
                    if urls:
                        todas_urls.extend(urls)
                        print(f"   ✅ {len(urls)} URLs encontradas")
                    else:
                        print(f"   ❌ Nenhuma URL encontrada")
                except Exception as e:
                    print(f"   ❌ Erro na fonte: {e}")
                
                time.sleep(1)  # Pausa entre fontes
        
        # Remover duplicatas
        urls_unicas = list(set(todas_urls))
        print(f"\n📊 RESUMO COLETA:")
        print(f"   URLs totais: {len(todas_urls)}")
        print(f"   URLs únicas: {len(urls_unicas)}")
        
        return urls_unicas

    def download_com_fallback(self, urls, pasta_destino, especie, quantidade_desejada):
        """Baixa imagens com sistema de fallback"""
        print(f"\n⬇️  INICIANDO DOWNLOADS...")
        
        # Contar imagens atuais
        arquivos_atuais = [f for f in os.listdir(pasta_destino) if f.endswith(('.jpg', '.jpeg', '.png'))]
        count_inicial = len(arquivos_atuais)
        contador = 0
        
        # Randomizar URLs para não pegar sempre as mesmas
        urls_randomizadas = urls.copy()
        random.shuffle(urls_randomizadas)
        
        for i, url in enumerate(urls_randomizadas):
            if contador >= quantidade_desejada:
                break
            
            numero_arquivo = count_inicial + contador + 1
            nome_arquivo = f"{especie}_{numero_arquivo:04d}.jpg"
            caminho_completo = os.path.join(pasta_destino, nome_arquivo)
            
            self.stats["total_tentativas"] += 1
            print(f"⬇️  [{contador+1}/{quantidade_desejada}] {nome_arquivo}: ", end="")
            
            if self.baixar_validar_imagem(url, caminho_completo):
                print("✅")
                contador += 1
                self.stats["sucessos"] += 1
            else:
                print("❌")
                self.stats["erros"] += 1
            
            time.sleep(0.8)  # Pausa entre downloads
        
        return contador

    def baixar_validar_imagem(self, url, caminho_destino):
        """Baixa e valida uma imagem com múltiplas tentativas"""
        max_tentativas = 3
        
        for tentativa in range(max_tentativas):
            try:
                response = self.session.get(url, timeout=20)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    
                    # Verificar se é imagem
                    if any(tipo in content_type for tipo in ['image/jpeg', 'image/png', 'image/jpg']):
                        
                        # Salvar temporário
                        temp_path = caminho_destino + '.tmp'
                        with open(temp_path, 'wb') as f:
                            f.write(response.content)
                        
                        # Validar
                        if self.validar_imagem(temp_path):
                            shutil.move(temp_path, caminho_destino)
                            return True
                        else:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                
            except Exception:
                if tentativa < max_tentativas - 1:
                    time.sleep(2)  # Pausa antes de tentar novamente
                
        return False

    def validar_imagem(self, caminho):
        """Validação rigorosa de imagem"""
        try:
            with Image.open(caminho) as img:
                # Verificar formato
                if img.format not in ['JPEG', 'PNG', 'JPG']:
                    return False
                
                # Verificar dimensões
                if img.size[0] < 150 or img.size[1] < 150:
                    return False
                
                # Verificar tamanho do arquivo
                if os.path.getsize(caminho) < 5000:  # 5KB mínimo
                    return False
                
                # Verificar se pode converter
                img.convert('RGB')
                return True
                
        except Exception:
            return False

def executar_coleta_robusta():
    """Execução principal com sistema robusto"""
    
    # Configuração das espécies
    ESPECIES_CONFIG = {
        "mammuthus_primigenius": [
            "mammuthus primigenius",
            "woolly mammoth", 
            "mammoth fossil",
            "mammoth skeleton",
            "pleistocene mammoth"
        ],
        "smilodon_fatalis": [
            "smilodon fatalis",
            "sabertooth tiger",
            "saber tooth cat", 
            "smilodon fossil",
            "ice age predator"
        ]
    }
    
    print("🦴 DATASET BUILDER - VERSÃO ROBUSTA")
    print("="*45)
    
    # Menu
    print(f"\n📋 OPÇÕES DE COLETA:")
    print(f"1. 🚀 Coleta rápida (50 por espécie)")
    print(f"2. 📈 Coleta média (100 por espécie)")
    print(f"3. 🎯 Coleta robusta (200 por espécie)")
    print(f"4. ⚙️  Quantidade personalizada")
    
    while True:
        opcao = input(f"\nEscolha (1-4): ").strip()
        if opcao in ['1', '2', '3', '4']:
            break
        print("❌ Opção inválida!")
    
    # Definir quantidade
    if opcao == "1":
        quantidade = 50
    elif opcao == "2":
        quantidade = 100
    elif opcao == "3":
        quantidade = 200
    else:
        while True:
            try:
                quantidade = int(input("Quantidade por espécie: "))
                if quantidade > 0:
                    break
                print("❌ Digite um número maior que 0!")
            except ValueError:
                print("❌ Digite apenas números!")
    
    # Confirmar
    print(f"\n📊 CONFIGURAÇÃO:")
    print(f"   Quantidade por espécie: {quantidade}")
    print(f"   Total de espécies: {len(ESPECIES_CONFIG)}")
    print(f"   Fontes: 5 diferentes")
    print(f"   Tempo estimado: {quantidade//5} minutos")
    
    if input(f"\n🚀 Iniciar coleta robusta? (s/n): ").lower() not in ['s', 'sim', 'y', 'yes']:
        print("❌ Cancelado.")
        return
    
    # Criar builder
    builder = RobustDatasetBuilder()
    
    # Executar para cada espécie
    resultados = {}
    
    for especie, termos in ESPECIES_CONFIG.items():
        print(f"\n" + "="*50)
        print(f"🦴 PROCESSANDO: {especie}")
        print("="*50)
        
        # Criar pasta se não existir
        pasta_destino = f"data/images/{especie}"
        os.makedirs(pasta_destino, exist_ok=True)
        
        # Coletar URLs
        urls = builder.coletar_urls_todas_fontes(especie, termos)
        
        if not urls:
            print(f"❌ NENHUMA URL ENCONTRADA PARA {especie}!")
            print(f"💡 SOLUÇÕES:")
            print(f"   1. Verifique sua conexão com internet")
            print(f"   2. Tente novamente em alguns minutos") 
            print(f"   3. Use coleta manual (vou te ensinar)")
            resultados[especie] = 0
            continue
        
        # Fazer downloads
        adicionadas = builder.download_com_fallback(urls, pasta_destino, especie, quantidade)
        resultados[especie] = adicionadas
    
    # Relatório final
    print(f"\n" + "="*50)
    print(f"📊 RELATÓRIO FINAL")
    print("="*50)
    
    total_adicionadas = 0
    for especie, quantidade_final in resultados.items():
        print(f"🦴 {especie}: {quantidade_final} imagens")
        total_adicionadas += quantidade_final
    
    print(f"\n📈 ESTATÍSTICAS:")
    print(f"   Tentativas: {builder.stats['total_tentativas']}")
    print(f"   ✅ Sucessos: {builder.stats['sucessos']}")
    print(f"   ❌ Erros: {builder.stats['erros']}")
    print(f"   Taxa de sucesso: {builder.stats['sucessos']/max(builder.stats['total_tentativas'],1)*100:.1f}%")
    
    if total_adicionadas == 0:
        print(f"\n❌ NENHUMA IMAGEM FOI COLETADA!")
        print(f"🆘 PLANO B - COLETA MANUAL:")
        print(f"   1. Acesse: https://commons.wikimedia.org")
        print(f"   2. Busque: 'mammoth fossil skeleton'")
        print(f"   3. Baixe manualmente para: data/images/mammuthus_primigenius/")
        print(f"   4. Repita para smilodon: 'saber tooth cat fossil'")
    else:
        print(f"\n✨ SUCESSO! {total_adicionadas} imagens coletadas")
        print(f"🎯 Próximos passos:")
        print(f"   1. python train.py")
        print(f"   2. python predictor.py")

if __name__ == "__main__":
    executar_coleta_robusta()