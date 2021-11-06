async function sendForm(url, game_id, method) {
    let gama_data = {
        game_id: game_id
    };

    await fetch(url, {
        method: method,
        headers: {
            'Content-Type': 'application/json;charset=utf-8',
            'X-CSRFToken': csrf_token
        },
        body: JSON.stringify(gama_data)
    });
}

async function changeAddButtonState() {
    let btnMustState = getMustButtonState('btn-must');
    let btnAdded = document.getElementById('btn-must-added');
    if (btnMustState) {
        btnAdded.style.display = 'inline-block';
    } else {
        btnAdded.style.display = 'none';
    }
}

function getMustButtonState(){
    let btnMustText = document.getElementById('btn-must').innerText;
        if (btnMustText == 'REMOVE') {
            return true;
        } else {
            return false;
        }

}

function changeMustButtonState(game_id) {
    let btnMustName = document.getElementById('btn-must');
    let btnMustState = getMustButtonState();
    if (btnMustState) {
        btnMustName.innerText = 'MUST';
        sendForm(url, game_id, 'DELETE');
        return  false;
    } else {
        btnMustName.innerText = 'REMOVE';
        sendForm(url, game_id, 'POST');
        return  true;
    }
}

function changeDeleteBlockState(id) {
    console.log('favourite_game_block' + String(id));
    let btnDelete = document.getElementById('favourite_game_block' + String(id));
        btnDelete.style.display = 'none';
}
