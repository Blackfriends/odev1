# CPU Scheduling Simulator — Ödev 1 (İşlemci Zamanlama)

Bu proje, verilen CSV girdilerine göre çeşitli CPU zamanlama algoritmalarını simüle eder ve her algoritma için zaman tabloları ve istatistikleri dosyalara yazar.

Desteklenen algoritmalar
- FCFS (First Come First Served)
- Preemptive SJF (SRTF)
- Non-Preemptive SJF
- Round Robin
- Preemptive Priority
- Non-Preemptive Priority

Kurallar / Varsayımlar
- Girdi CSV'si başlık satırı içermelidir: `pid,arrival,burst,priority` (priority isteğe bağlı, varsayılan 0)
- `arrival` ve `burst` sayısal (tam veya ondalık) olarak beklenir.
- Bağlam değiştirme süresi sabit: 0.001 birim zaman.

Çıktılar
Her algoritma için `out/<ALG>` klasörü oluşturulur ve içinde:
- `timeline.txt` — zaman tablosu (örneğe uygun formatta)
- `stats.txt` — bekleme, tamamlanma, throughput, CPU verimliliği vb.
- `processes.csv` — süreçlere ait tamamlanma zamanları

Kullanım
1. Örnek veriler workspace içinde `case1.csv` ve `case2.csv` olarak eklidir.
2. Simülasyonu çalıştırmak için:

```bash
python3 scheduler.py --input case1.csv --output out_case1
```

3. Round Robin quantum'u değiştirmek için `--quantum` kullanın (varsayılan 4):

```bash
python3 scheduler.py -i case1.csv -o out_case1 -q 3
```

4. Bonus (her algoritmayı ayrı thread'te çalıştırmak için):

```bash
python3 scheduler.py -i case1.csv -o out_case1 --threads
```

Notlar
- `--threads` aktifken tüm algoritmalar aynı girdiye göre eş zamanlı çalışır (bonus +70 puan).
- Proje tesliminde rapor ve kullanıcı kılavuzu olarak bu `README.md` dosyasını düzenleyip genişletin.

Geliştirme ve test önerileri
- `report_generator.py` gibi bir script ile her algoritmanın `stats.txt` çıktısını toplayıp birleştirerek rapor (PDF/HTML) oluşturabilirsiniz.
