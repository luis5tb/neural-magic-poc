```
curl 'http://localhost:8080/v2/models/predict/infer' \
    -XPOST \
    -H 'Content-type: application/json' \
    -d '{"sequences": ["Snorlax loves my Tesla!"]}'
```
